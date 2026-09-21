"""
Depth growth for a GPT that is already mid-training.

The point of this file is to insert transformer blocks into a model *without
changing the function it computes*, so that an RL run can be interrupted, made
deeper, and resumed with the reward curve continuous across the seam. Anything
that does move the function is a bug, and `capture_logits` exists so the training
loop can assert that at runtime.

Why that is nearly free here: a Block's only contributions to the residual stream
are `attn.c_proj(...)` and `mlp.c_proj(...)` (see Block.forward). Zero both and the
block is exactly the identity map. This is the same trick as LLaMA Pro's block
expansion (arXiv:2401.02415), and nanochat's own init already does it for c_proj.

What is NOT free, and is handled below:

  * `attn.layer_idx` indexes the KV cache. Inserting a block mid-stack renumbers
    every block after it; miss this and inference reads the wrong cache slots and
    never advances position.
  * `has_ve(i, n_layer)` keys off n_layer, so re-deriving the value-embedding set
    at a new depth can invert it for every already-trained layer. We remap the
    existing assignment instead and pin it in the config.
  * `backout_layer = n_layer // 2` moves, and the final hidden state is
    `x - backout_lambda * x_backout` with backout_lambda ~0.5 in a trained model.
    We pin the tap so it keeps capturing the same block's output.
  * `resid_lambdas` / `x0_lambdas` are `(n_layer,)` tensors that must be replaced,
    which orphans their AdamW moments unless they are migrated.
  * Matrix optimizer state can be positionally indexed (Muon's is a
    `(chunk_size, *shape)` stack), so appending to an existing group would
    misalign it against parameters. New parameters therefore go into *new*
    parameter groups, which also happens to be exactly where a separate learning
    rate belongs (see §"learning rate" below).

Learning rate: at nanochat's RL defaults (matrix_lr 0.02 x init_lr_frac 0.05 = 1e-3,
decayed linearly to zero) a Muon step moves a matrix by ~1e-3 in spectral norm,
while a trained c_proj in this model measures ~5. A layer grown at the midpoint of
a 280-step run would reach well under 1% of that, i.e. it would never leave
identity and the experiment would measure nothing. `lr_mult` gives the new groups
their own, much larger learning rate; `warmup` ramps it so Muon's first
(gradient-magnitude-independent) step does not hit a zero matrix at full size.

Single rank only. With world_size > 1, growing a group changes
`chunk_size = ceil(K/N)` and shifts every rank's ownership boundary, which would
require resharding the momentum stack.
"""

import torch
import torch.nn as nn

from nanochat.gpt import Block, resolve_ve_layers, resolve_backout_layer
from nanochat.common import print0

GROW_INITS = ("copy", "fresh", "random")


def parse_schedule(spec, num_steps):
    """"0.33,0.66" -> [92, 185] for num_steps=280.

    A value below 1.0 is a fraction of the horizon, so a schedule stays portable
    across horizons; anything else must be a whole number and is an absolute step.
    Values like 1.5 are ambiguous under both readings and are rejected rather than
    silently truncated. Step 0 is allowed (the "grow immediately" control arm).
    """
    if not spec:
        return []
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        assert v >= 0, f"growth point {tok!r} must be non-negative"
        if v < 1.0:
            step = int(round(v * num_steps))
        else:
            assert v == int(v), f"growth point {tok!r} is neither a fraction (<1) nor a whole step"
            step = int(v)
        assert 0 <= step < num_steps, f"growth step {step} outside [0, {num_steps})"
        out.append(step)
    return sorted(out)


def _insertion_index(n_layer, position):
    if position == "middle":
        # "Is One Layer Enough?" (arXiv:2607.01232) finds RL's high-contribution
        # layers concentrate mid-stack, so that is where new capacity belongs.
        return n_layer // 2
    if position == "end":
        return n_layer
    if position == "start":
        return 0
    at = int(position)
    assert 0 <= at <= n_layer, f"insertion index {at} outside [0, {n_layer}]"
    return at


@torch.no_grad()
def _init_block(block, config, src, mode, generator=None):
    """Fill a newly constructed block. `copy`/`fresh` are identity; `random` is not."""
    assert mode in GROW_INITS, f"unknown grow init {mode!r}, expected one of {GROW_INITS}"
    n_embd = config.n_embd
    s = 3 ** 0.5 * n_embd ** -0.5                 # matches GPT.init_weights
    s_mlp = 3 ** 0.5 * (4 * n_embd) ** -0.5       # c_proj reads 4*n_embd inputs

    def unif(t, a, b):
        if generator is None:
            t.uniform_(a, b)
        else:
            t.uniform_(a, b, generator=generator)

    if mode == "copy":
        # LLaMA Pro: inherit the neighbour's features so the block does something
        # sensible the moment c_proj lifts off zero, rather than projecting noise.
        block.attn.c_q.weight.copy_(src.attn.c_q.weight)
        block.attn.c_k.weight.copy_(src.attn.c_k.weight)
        block.attn.c_v.weight.copy_(src.attn.c_v.weight)
        block.mlp.c_fc.weight.copy_(src.mlp.c_fc.weight)
        if block.attn.ve_gate is not None:
            if src.attn.ve_gate is not None:
                block.attn.ve_gate.weight.copy_(src.attn.ve_gate.weight)
            else:
                unif(block.attn.ve_gate.weight, 0.0, 0.02)
    else:
        unif(block.attn.c_q.weight, -s, s)
        unif(block.attn.c_k.weight, -s, s)
        unif(block.attn.c_v.weight, -s, s)
        unif(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
        if block.attn.ve_gate is not None:
            unif(block.attn.ve_gate.weight, 0.0, 0.02)

    if mode == "random":
        # Deliberately NOT function preserving: this is the arm that tests the
        # "reward dips, then recovers" hypothesis directly.
        unif(block.attn.c_proj.weight, -s, s)
        unif(block.mlp.c_proj.weight, -s_mlp, s_mlp)
    else:
        block.attn.c_proj.weight.zero_()
        block.mlp.c_proj.weight.zero_()


def _grow_vector(vec, at, k, fill):
    out = torch.empty(vec.shape[0] + k, dtype=vec.dtype, device=vec.device)
    out[:at] = vec[:at]
    out[at:at + k] = fill
    out[at + k:] = vec[at:]
    return out


def _swap_optimizer_param(optimizer, old, new, at, k):
    """Replace a param in whatever group holds it, zero-padding its moments to match."""
    for group in optimizer.param_groups:
        for i, p in enumerate(group["params"]):
            if p is old:
                group["params"][i] = new
                state = optimizer.state.pop(old, None)
                if state:
                    migrated = {}
                    for key, v in state.items():
                        if torch.is_tensor(v) and tuple(v.shape) == tuple(old.shape):
                            migrated[key] = _grow_vector(v, at, k, 0.0)
                        else:
                            migrated[key] = v   # 'step' counters and the like
                    optimizer.state[new] = migrated
                return True
    return False


@torch.no_grad()
def grow_depth(model, optimizer, num_layers, position="middle", init="copy",
               add_ve=False, lr_mult=50.0, warmup=5, step=0, generator=None):
    """Insert `num_layers` blocks and extend `optimizer` to cover them, in place.

    Returns a dict of what happened, suitable for logging.
    """
    cfg = model.config
    k = int(num_layers)
    assert k > 0, "num_layers must be positive"
    old_n = cfg.n_layer
    new_n = old_n + k
    at = _insertion_index(old_n, position)
    device = model.get_device()
    src_idx = max(at - 1, 0)
    src = model.transformer.h[src_idx]
    dtype = src.mlp.c_fc.weight.dtype

    remap = lambda i: i if i < at else i + k

    # --- structure that must survive the depth change ------------------------
    new_ve = {remap(i) for i in model.ve_layer_set}
    if add_ve:
        new_ve |= {at + j for j in range(k)}
    new_backout = remap(model.backout_layer)
    long_window = cfg.sequence_len
    old_chars = "".join("L" if w == long_window else "S" for w, _ in model.window_sizes)
    new_chars = old_chars[:at] + old_chars[src_idx] * k + old_chars[at:]

    # --- build and splice in the new blocks ----------------------------------
    # Built and initialised on the generator's device, then moved: a CPU generator
    # cannot fill an MPS/CUDA tensor, and seeding growth is what makes the `random`
    # and `fresh` arms reproducible across seeds.
    init_device = generator.device if generator is not None else torch.device("cpu")
    blocks = []
    for j in range(k):
        b = Block(cfg, at + j, use_ve=((at + j) in new_ve)).to(device=init_device, dtype=dtype)
        _init_block(b, cfg, src, init, generator=generator)   # copy_ handles the cross-device read
        blocks.append(b.to(device=device))
    h = model.transformer.h
    model.transformer.h = nn.ModuleList(list(h[:at]) + blocks + list(h[at:]))
    # layer_idx indexes the KV cache, so every block after the seam is renumbered.
    for i, b in enumerate(model.transformer.h):
        b.attn.layer_idx = i

    # --- per-layer scalars: identity for the new entries ---------------------
    # Not the init_weights formula (1.15 -> 1.05): the new block is only the
    # identity if its residual scale is exactly 1.0 and its x0 blend exactly 0.0.
    old_resid, old_x0 = model.resid_lambdas, model.x0_lambdas
    model.resid_lambdas = nn.Parameter(_grow_vector(old_resid.data, at, k, 1.0))
    model.x0_lambdas = nn.Parameter(_grow_vector(old_x0.data, at, k, 0.0))

    # --- value embeddings: remap keys, never rebuild from the formula ---------
    kv_dim = cfg.n_kv_head * (cfg.n_embd // cfg.n_head)
    ve_dtype = next(iter(model.value_embeds.parameters())).dtype if len(model.value_embeds) else dtype
    remapped = nn.ModuleDict()
    for key, emb in model.value_embeds.items():
        remapped[str(remap(int(key)))] = emb      # same module object: state and optimizer entry preserved
    new_ve_params = []
    if add_ve:
        s = 3 ** 0.5 * cfg.n_embd ** -0.5
        for j in range(k):
            if (at + j) in new_ve:
                e = nn.Embedding(model.padded_vocab_size, kv_dim).to(device=init_device, dtype=ve_dtype)
                if generator is None:
                    e.weight.uniform_(-s, s)
                else:
                    e.weight.uniform_(-s, s, generator=generator)
                e = e.to(device=device)
                remapped[str(at + j)] = e
                new_ve_params.append(e.weight)
    model.value_embeds = remapped

    # --- commit the config, then re-derive everything that reads it ----------
    cfg.n_layer = new_n
    cfg.ve_layers = tuple(sorted(new_ve))
    cfg.backout_layer = int(new_backout)
    cfg.window_layers = new_chars
    model.ve_layer_set = set(new_ve)
    model.backout_layer = int(new_backout)
    model.window_sizes = model._compute_window_sizes(cfg)

    # --- optimizer: new params into NEW groups -------------------------------
    adamw_tmpl = next(g for g in optimizer.param_groups if g["kind"] == "adamw")
    # A grown layer inherits the matrix optimizer of the block immediately to its
    # left (or block zero when prepending). This makes a growth run compatible
    # with a depth-wise Muon/AdamW/Sophia layout without guessing a new policy.
    source_matrix_param = src.attn.c_q.weight
    matrix_tmpl = next(
        g for g in optimizer.param_groups
        if any(p is source_matrix_param for p in g["params"])
    )
    matrix_kind = matrix_tmpl["kind"]
    new_lr = matrix_tmpl["initial_lr"] * lr_mult
    added = 0
    by_shape = {}
    for b in blocks:
        for p in b.parameters():
            by_shape.setdefault(tuple(p.shape), []).append(p)
    for shape in sorted(by_shape):
        common = dict(
            kind=matrix_kind, params=by_shape[shape], lr=new_lr, initial_lr=new_lr,
            is_matrix=True, layer_indices=tuple(range(at, at + k)),
            weight_decay=matrix_tmpl["weight_decay"],
            grown_at=int(step), grow_warmup=int(warmup), grow_tag="new_matrix",
        )
        if matrix_kind == "muon":
            optimizer.add_param_group(dict(
                **common, momentum=matrix_tmpl["momentum"], ns_steps=matrix_tmpl["ns_steps"],
                beta2=matrix_tmpl["beta2"],
            ))
        elif matrix_kind == "adamw":
            optimizer.add_param_group(dict(
                **common, betas=matrix_tmpl["betas"], eps=matrix_tmpl["eps"],
            ))
        elif matrix_kind == "sophia":
            optimizer.add_param_group(dict(
                **common, betas=matrix_tmpl["betas"], rho=matrix_tmpl["rho"],
                batch_size=matrix_tmpl["batch_size"],
                hessian_update_interval=matrix_tmpl["hessian_update_interval"],
                eps=matrix_tmpl["eps"],
            ))
        else:
            raise AssertionError(f"Unknown matrix optimizer kind {matrix_kind!r}")
        added += len(by_shape[shape])
    if new_ve_params:
        ve_lr = adamw_tmpl["initial_lr"] * lr_mult
        optimizer.add_param_group(dict(
            kind="adamw", params=new_ve_params, lr=ve_lr, initial_lr=ve_lr,
            betas=adamw_tmpl["betas"], eps=adamw_tmpl["eps"], weight_decay=adamw_tmpl["weight_decay"],
            grown_at=int(step), grow_warmup=int(warmup), grow_tag="new_ve",
        ))
        added += len(new_ve_params)
    # The lambda vectors are new tensors, so their moments have to move with them.
    assert _swap_optimizer_param(optimizer, old_resid, model.resid_lambdas, at, k), "resid_lambdas not found in optimizer"
    assert _swap_optimizer_param(optimizer, old_x0, model.x0_lambdas, at, k), "x0_lambdas not found in optimizer"

    info = {
        "step": int(step), "at": at, "k": k, "init": init, "position": str(position),
        "n_layer_before": old_n, "n_layer_after": new_n,
        "backout_layer": int(new_backout), "ve_layers": tuple(sorted(new_ve)),
        "window_layers": new_chars, "new_param_tensors": added,
        "new_params": int(sum(p.numel() for b in blocks for p in b.parameters())
                          + sum(p.numel() for p in new_ve_params)),
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "new_lr": float(new_lr), "matrix_optimizer": matrix_kind,
        "lr_mult": float(lr_mult), "warmup": int(warmup),
    }
    print0(f"[grow] step {step}: {old_n} -> {new_n} layers, inserted {k} at index {at} "
           f"(init={init}, ve={'on' if add_ve else 'off'}), "
           f"+{info['new_params']/1e6:.1f}M params -> {info['total_params']/1e6:.1f}M total, "
           f"new {matrix_kind} group lr={new_lr:.2e} (x{lr_mult}), backout tap -> layer {new_backout}")
    return info


def set_learning_rates(optimizer, step, lrm):
    """Apply the global schedule, plus a per-event warmup for groups added by growth.

    Groups without a `grown_at` key behave exactly as the original two-line loop did.
    """
    for group in optimizer.param_groups:
        m = lrm
        born = group.get("grown_at")
        if born is not None:
            w = int(group.get("grow_warmup", 0))
            if w > 0:
                m = m * min(1.0, (step - born + 1) / w)
        group["lr"] = group["initial_lr"] * m


@torch.no_grad()
def capture_logits(model, ids):
    """Logits on a fixed probe batch. Used to verify growth really was identity."""
    was_training = model.training
    model.eval()
    out = model(ids).float().clone()
    if was_training:
        model.train()
    return out


def new_layer_indices(info_list, n_layer):
    """Which current indices correspond to layers added by growth, for per-layer logging."""
    grown = set()
    for info in info_list:
        at, k = info["at"], info["k"]
        grown = {i if i < at else i + k for i in grown} | set(range(at, at + k))
    return sorted(i for i in grown if i < n_layer)
