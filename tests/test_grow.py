"""
Tests for nanochat/grow.py.

The load-bearing test is test_growth_is_identity: growth that preserves the
function is the whole premise, and a single logits comparison catches every one
of the silent breakages (value-embedding parity, backout tap, window pattern,
KV-cache layer_idx renumbering) at once.
"""

import copy
import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.grow import (
    grow_depth, set_learning_rates, parse_schedule, capture_logits, new_layer_indices,
)


def make_model(n_layer=6, window_pattern="SSSL", seed=0):
    torch.manual_seed(seed)
    cfg = GPTConfig(sequence_len=1024, vocab_size=128, n_layer=n_layer,
                    n_head=2, n_kv_head=2, n_embd=64, window_pattern=window_pattern)
    model = GPT(cfg)
    model.init_weights()
    # Nudge the trained-model quantities off their init values: a model straight
    # out of init_weights has resid_lambdas near 1 and backout_lambda at 0.2, which
    # would hide exactly the bugs these tests exist to catch.
    with torch.no_grad():
        model.resid_lambdas.uniform_(0.5, 1.1)
        model.x0_lambdas.uniform_(-1.0, 3.0)
        model.backout_lambda.fill_(0.49)
        for b in model.transformer.h:
            b.attn.c_proj.weight.uniform_(-0.1, 0.1)
            b.mlp.c_proj.weight.uniform_(-0.1, 0.1)
    model.eval()
    return model


def probe(model, seed=1234, B=2, T=16):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.config.vocab_size, (B, T), generator=g)
    return ids


def fresh_optimizer(model):
    opt = model.setup_optimizer()
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
    return opt


def test_depth_memory_attention_runs_when_enabled():
    cfg = GPTConfig(sequence_len=64, vocab_size=128, n_layer=4,
                    n_head=2, n_kv_head=2, n_embd=32,
                    window_pattern="L", depth_memory_layers=2)
    model = GPT(cfg)
    model.init_weights()
    ids = probe(model, seed=321, B=1, T=12)
    out = model(ids)
    assert out.shape == (1, 12, 128)


# --------------------------------------------------------------------------- #
# function preservation

@pytest.mark.parametrize("position", ["middle", "end", "start", 2])
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("init", ["copy", "fresh"])
def test_growth_is_identity(position, k, init):
    model = make_model()
    ids = probe(model)
    before = capture_logits(model, ids)
    grow_depth(model, fresh_optimizer(model), k, position=position, init=init)
    after = capture_logits(model, ids)
    delta = (after - before).abs().max().item()
    assert delta < 1e-5, f"growth changed the function: max|dlogit| = {delta}"


@pytest.mark.parametrize("window_pattern", ["L", "SSSL", "SL"])
def test_growth_is_identity_across_window_patterns(window_pattern):
    """Guards the final-layer override: window_sizes[-1] is forced long, so an
    end-append would otherwise let the old last layer revert to its pattern char."""
    model = make_model(window_pattern=window_pattern)
    ids = probe(model)
    before = capture_logits(model, ids)
    grow_depth(model, fresh_optimizer(model), 2, position="end")
    after = capture_logits(model, ids)
    assert (after - before).abs().max().item() < 1e-5


def test_odd_growth_preserves_value_embedding_assignment():
    """has_ve keys off n_layer, so an odd depth change inverts it for every layer.
    Pinning ve_layers is what makes odd growth safe."""
    model = make_model(n_layer=6)
    before_ve = set(model.ve_layer_set)
    ids = probe(model)
    logits_before = capture_logits(model, ids)
    grow_depth(model, fresh_optimizer(model), 1, position="middle")
    # every previously-VE layer is still VE, at its shifted index
    assert {i if i < 3 else i + 1 for i in before_ve} == set(model.ve_layer_set)
    assert (capture_logits(model, ids) - logits_before).abs().max().item() < 1e-5


def test_backout_tap_follows_its_layer():
    model = make_model(n_layer=6)          # backout_layer = 3
    assert model.backout_layer == 3
    grow_depth(model, fresh_optimizer(model), 2, position="middle")   # insert at 3
    # old layer 3 now sits at index 5; the tap must have moved with it, NOT to 8//2=4
    assert model.backout_layer == 5
    assert model.config.backout_layer == 5


def test_random_init_is_not_identity():
    """The arm that deliberately perturbs the policy must actually perturb it."""
    model = make_model()
    ids = probe(model)
    before = capture_logits(model, ids)
    grow_depth(model, fresh_optimizer(model), 2, position="middle", init="random")
    after = capture_logits(model, ids)
    assert (after - before).abs().max().item() > 1e-3


def test_layer_idx_is_renumbered():
    """attn.layer_idx indexes the KV cache; a stale one silently corrupts inference."""
    model = make_model(n_layer=6)
    grow_depth(model, fresh_optimizer(model), 2, position="middle")
    assert [b.attn.layer_idx for b in model.transformer.h] == list(range(8))


def test_value_embeddings_can_be_added():
    model = make_model(n_layer=6)
    n_before = len(model.value_embeds)
    ids = probe(model)
    before = capture_logits(model, ids)
    grow_depth(model, fresh_optimizer(model), 2, position="middle", add_ve=True)
    assert len(model.value_embeds) == n_before + 2
    # a new VE still sits behind a zeroed c_proj, so the block remains the identity
    assert (capture_logits(model, ids) - before).abs().max().item() < 1e-5


# --------------------------------------------------------------------------- #
# checkpoint round-trip

def test_grown_model_roundtrips_through_config():
    """A grown checkpoint rebuilt from meta['model_config'] must be the same function.
    Without pinned ve_layers/backout_layer/window_layers it would reload as a
    different model while loading cleanly with strict=True."""
    model = make_model(n_layer=6, window_pattern="SSSL")
    ids = probe(model)
    grow_depth(model, fresh_optimizer(model), 3, position="middle")
    expected = capture_logits(model, ids)

    import json
    config_kwargs = json.loads(json.dumps(model.config.__dict__))   # force the JSON round-trip
    rebuilt = GPT(GPTConfig(**config_kwargs))
    rebuilt.init_weights()
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    rebuilt.eval()
    assert (capture_logits(rebuilt, ids) - expected).abs().max().item() < 1e-5


def test_old_checkpoints_build_unchanged():
    """Configs written before these fields existed must build the identical model."""
    old_kwargs = dict(sequence_len=1024, vocab_size=128, n_layer=6,
                      n_head=2, n_kv_head=2, n_embd=64, window_pattern="SSSL")
    cfg = GPTConfig(**old_kwargs)
    model = GPT(cfg)
    assert model.backout_layer == 3
    assert model.ve_layer_set == {1, 3, 5}
    assert model.window_sizes == GPT(GPTConfig(**old_kwargs)).window_sizes
    assert model.window_sizes[-1][0] == cfg.sequence_len   # final layer still forced long


# --------------------------------------------------------------------------- #
# optimizer

def test_existing_optimizer_state_is_untouched():
    """Rebuilding the optimizer after growth would reset every moment, a far bigger
    perturbation than the growth itself. Growth must leave old state alone."""
    model = make_model(n_layer=6)
    model.train()
    opt = fresh_optimizer(model)
    ids = probe(model)
    model(ids, ids.clone()).backward()
    opt.step()
    model.zero_grad(set_to_none=True)

    tracked = {}
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p)
            if st and "exp_avg" in st:
                tracked[id(p)] = st["exp_avg"].clone()
    assert tracked, "expected AdamW moments to exist after one step"

    grow_depth(model, opt, 2, position="middle")

    survived = 0
    for group in opt.param_groups:
        for p in group["params"]:
            if id(p) in tracked:
                assert torch.equal(opt.state[p]["exp_avg"], tracked[id(p)])
                survived += 1
    assert survived >= len(tracked) - 2, "only the two lambda vectors may be replaced"


def test_lambda_moments_are_migrated():
    model = make_model(n_layer=6)
    model.train()
    opt = fresh_optimizer(model)
    ids = probe(model)
    model(ids, ids.clone()).backward()
    opt.step()
    model.zero_grad(set_to_none=True)
    old_resid_moment = opt.state[model.resid_lambdas]["exp_avg"].clone()

    grow_depth(model, opt, 2, position="middle")

    new_moment = opt.state[model.resid_lambdas]["exp_avg"]
    assert new_moment.shape == (8,)
    assert torch.equal(new_moment[:3], old_resid_moment[:3])       # before the seam
    assert torch.equal(new_moment[5:], old_resid_moment[3:])       # after the seam
    assert torch.all(new_moment[3:5] == 0)                          # the new entries


def test_new_params_are_trainable_after_growth():
    """The whole experiment is void if the grown layers cannot move."""
    model = make_model(n_layer=6)
    model.train()
    opt = fresh_optimizer(model)
    grow_depth(model, opt, 2, position="middle", lr_mult=50.0, warmup=0, step=0)
    new_block = model.transformer.h[3]
    assert new_block.mlp.c_proj.weight.abs().max().item() == 0.0

    ids = probe(model)
    model(ids, ids.clone()).backward()
    opt.step()
    assert new_block.mlp.c_proj.weight.abs().max().item() > 0.0, "grown layer did not move"


def test_growth_adds_param_groups_not_params_to_old_groups():
    """Muon state is a positionally indexed stack; appending to an existing group
    would misalign momentum rows against parameters."""
    model = make_model(n_layer=6)
    opt = fresh_optimizer(model)
    sizes_before = [len(g["params"]) for g in opt.param_groups]
    n_groups_before = len(opt.param_groups)
    grow_depth(model, opt, 2, position="middle")
    assert len(opt.param_groups) > n_groups_before
    assert [len(g["params"]) for g in opt.param_groups[:n_groups_before]] == sizes_before


# --------------------------------------------------------------------------- #
# learning rate plumbing

def test_set_learning_rates_matches_old_behaviour_without_growth():
    model = make_model()
    opt = fresh_optimizer(model)
    expected = [g["initial_lr"] * 0.3 for g in opt.param_groups]
    set_learning_rates(opt, step=17, lrm=0.3)
    assert [g["lr"] for g in opt.param_groups] == expected


def test_grown_groups_warm_up():
    model = make_model()
    opt = fresh_optimizer(model)
    grow_depth(model, opt, 2, position="middle", lr_mult=50.0, warmup=5, step=100)
    grown = [g for g in opt.param_groups if g.get("grow_tag") == "new_matrix"]
    old = [g for g in opt.param_groups if "grown_at" not in g]
    assert grown

    for offset, frac in ((0, 1 / 5), (2, 3 / 5), (4, 1.0), (50, 1.0)):
        set_learning_rates(opt, step=100 + offset, lrm=1.0)
        for g in grown:
            assert g["lr"] == pytest.approx(g["initial_lr"] * frac)
        for g in old:
            assert g["lr"] == pytest.approx(g["initial_lr"])


def test_grown_group_lr_is_larger_than_base():
    """At the base RL learning rate a grown layer cannot leave identity; the
    multiplier is what makes the arm measure anything at all."""
    model = make_model()
    opt = fresh_optimizer(model)
    base = next(g["initial_lr"] for g in opt.param_groups if g["kind"] == "muon")
    grow_depth(model, opt, 2, lr_mult=50.0)
    grown = next(g for g in opt.param_groups if g.get("grow_tag") == "new_matrix")
    assert grown["initial_lr"] == pytest.approx(base * 50.0)


# --------------------------------------------------------------------------- #
# schedule helpers

def test_parse_schedule():
    assert parse_schedule("", 280) == []
    assert parse_schedule(None, 280) == []
    assert parse_schedule("0.33", 280) == [92]
    assert parse_schedule("0.33,0.66", 280) == [92, 185]
    assert parse_schedule("0", 280) == [0]
    assert parse_schedule("50,10", 280) == [10, 50]      # absolute, sorted
    with pytest.raises(AssertionError):
        parse_schedule("1.5", 280)                        # 420 >= 280


def test_new_layer_indices_tracks_successive_growth():
    model = make_model(n_layer=6)
    opt = fresh_optimizer(model)
    infos = [grow_depth(model, opt, 2, position="middle")]     # insert at 3 -> {3,4}
    assert new_layer_indices(infos, model.config.n_layer) == [3, 4]
    infos.append(grow_depth(model, opt, 2, position="middle"))  # n=8, insert at 4
    # the first pair shifts: index 3 stays, index 4 -> 6; the new pair is {4,5}
    assert new_layer_indices(infos, model.config.n_layer) == [3, 4, 5, 6]


def test_growth_accepts_a_generator_for_every_init_mode():
    """Seeded growth: new blocks are initialised on the generator's device and then
    moved, because a CPU generator cannot fill a tensor on an accelerator."""
    for init in ("copy", "fresh", "random"):
        model = make_model(n_layer=6)
        g = torch.Generator(device="cpu").manual_seed(11)
        grow_depth(model, fresh_optimizer(model), 2, position="middle", init=init,
                   add_ve=True, generator=g)
        target = model.transformer.h[0].mlp.c_fc.weight.device
        for p in model.transformer.h[3].parameters():
            assert p.device == target, f"{init}: grown block landed on {p.device}, not {target}"
        for p in model.value_embeds.parameters():
            assert p.device == target


def test_seeded_growth_is_reproducible():
    outs = []
    for _ in range(2):
        model = make_model(n_layer=6)
        g = torch.Generator(device="cpu").manual_seed(5)
        grow_depth(model, fresh_optimizer(model), 2, position="middle", init="random", generator=g)
        outs.append(model.transformer.h[3].mlp.c_proj.weight.clone())
    assert torch.equal(outs[0], outs[1])
