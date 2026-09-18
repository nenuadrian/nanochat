"""Per-prompt rollout budget allocation for RL fine-tuning.

Three strategies, all answering the same question: given a total budget of C
rollouts across B prompts, how many should each prompt get?

    uniform  n_q = C/B. What chat_rl did before, and the baseline.

    gvm      GVM, arXiv:2505.02391 Proposition 1.
             n_q proportional to G_q / sqrt(p_q + alpha / p_q^(beta-1)),
             with p_q the accept rate and G_q the mean gradient norm, both
             MEASURED by a pilot pass of N' rollouts per prompt.

    vip      VIP, ICLR 2026 (arXiv:2602.01601) Theorem 5.1.
             Minimises the Dr. GRPO gradient variance sum_q a_q (n_q-1)/n_q^2
             subject to sum_q n_q = C and L <= n_q <= U, with
             a_q = 4 sigma_q^2 p_q (1 - p_q) and p_q PREDICTED by a GP
             (see vip_gp.py) rather than measured.

The practical difference is where p_q comes from. GVM buys it with N' extra
rollouts per prompt per iteration, which is real compute that the budget C does
not account for; VIP predicts it for free but can be wrong. `rollout_accounting`
below exists to keep that comparison honest.
"""

import numpy as np

__all__ = [
    "uniform_allocation",
    "gvm_allocation",
    "vip_allocation",
    "round_preserving_sum",
    "allocation_stats",
    "rollout_accounting",
]


def round_preserving_sum(x, total, lo=0, hi=None):
    """Round reals to ints summing exactly to `total`, respecting [lo, hi]."""
    x = np.asarray(x, dtype=float)
    n = np.round(x).astype(int)
    n = np.clip(n, lo, hi if hi is not None else np.iinfo(np.int64).max)
    # Hand out (or claw back) the residual one unit at a time, preferring the
    # entries whose rounding error was largest, and never leaving [lo, hi].
    resid = x - n
    while n.sum() != total:
        err = total - n.sum()
        step = 1 if err > 0 else -1
        room = (n < hi) if (step > 0 and hi is not None) else (n > lo) if step < 0 else np.ones_like(n, bool)
        if not room.any():
            raise ValueError(f"cannot reach total={total} within [{lo},{hi}] for {len(n)} prompts")
        order = np.argsort(-resid if step > 0 else resid)
        for i in order:
            if room[i]:
                n[i] += step
                resid[i] -= step
                break
    return n


def uniform_allocation(num_prompts, budget, lo=0, hi=None):
    share = np.full(num_prompts, budget / num_prompts, dtype=float)
    return round_preserving_sum(share, budget, lo=lo, hi=hi)


def gvm_allocation(p, G, budget, alpha=1e-3, beta=2.0, lo=0, hi=None):
    """GVM Proposition 1.

    p, G are the measured accept rate and mean gradient norm per prompt. A prompt
    with p=0 has produced no correct rollout, so there is nothing to estimate and
    it gets nothing; same when G=0.
    """
    p = np.asarray(p, dtype=float)
    G = np.asarray(G, dtype=float)
    safe_p = np.where(p > 0, p, 1.0)  # placeholder; masked out below
    ratio = G / np.sqrt(safe_p + alpha / np.power(safe_p, beta - 1.0))
    ratio = np.where((p <= 0) | (G <= 0), 0.0, ratio)
    if ratio.sum() <= 0:
        return uniform_allocation(len(p), budget, lo=lo, hi=hi)
    return round_preserving_sum(ratio / ratio.sum() * budget, budget, lo=lo, hi=hi)


def _vip_n_of_lambda(a, lam, lo, hi):
    """Per-prompt n*(lambda) from VIP Theorem 5.1 (Dr. GRPO branch).

    Stationarity of a(n-1)/n^2 gives lambda = a (n-2)/n^3, whose right-hand side
    is strictly decreasing in n for n > 3 -- hence the paper's requirement L >= 3
    and the bisection below.
    """
    f = lambda n: a * (n - 2.0) / n**3
    if a <= 0:            # no variance to reduce; park it at the floor
        return float(lo)
    if lam <= f(hi):
        return float(hi)
    if lam >= f(lo):
        return float(lo)
    n_lo, n_hi = float(lo), float(hi)
    for _ in range(60):
        mid = 0.5 * (n_lo + n_hi)
        if f(mid) > lam:  # f decreasing: need a larger n
            n_lo = mid
        else:
            n_hi = mid
    return 0.5 * (n_lo + n_hi)


def vip_allocation(p_hat, budget, lo=3, hi=None, sigma=None):
    """VIP Theorem 5.1: allocate by bisection on the KKT multiplier.

    `sigma` is the per-prompt gradient-norm scale sigma_Z_q; when unknown it is
    taken as constant, which leaves the allocation driven purely by p(1-p) --
    the form the paper's ablation uses when no gradient-norm estimate is at hand.
    """
    p_hat = np.asarray(p_hat, dtype=float)
    B = len(p_hat)
    hi = budget if hi is None else hi
    if not (B * lo <= budget <= B * hi):
        raise ValueError(f"need B*L <= C <= B*U: {B}*{lo} <= {budget} <= {B}*{hi}")
    sig = np.ones(B) if sigma is None else np.asarray(sigma, dtype=float)
    a = 4.0 * sig**2 * p_hat * (1.0 - p_hat)

    # lambda -> total is non-increasing, so bisect on lambda to hit the budget.
    lam_hi = max(float(np.max(a * (lo - 2.0) / lo**3)), 1e-12)
    lam_lo = 0.0
    for _ in range(80):
        lam = 0.5 * (lam_lo + lam_hi)
        total = sum(_vip_n_of_lambda(ai, lam, lo, hi) for ai in a)
        if total > budget:
            lam_lo = lam
        else:
            lam_hi = lam
    n_real = np.array([_vip_n_of_lambda(ai, 0.5 * (lam_lo + lam_hi), lo, hi) for ai in a])
    return round_preserving_sum(n_real, budget, lo=lo, hi=hi)


def allocation_stats(n, p=None, prefix="alloc"):
    """Metrics that say whether the allocator is doing anything at all.

    If `conc` is near 1/B the allocation is effectively uniform; if `p_spread` is
    near zero there is no difference in difficulty to allocate on, and any
    allocator will look the same regardless of how good it is.
    """
    n = np.asarray(n, dtype=float)
    tot = max(n.sum(), 1.0)
    share = n / tot
    out = {
        f"{prefix}/total": float(n.sum()),
        f"{prefix}/min": float(n.min()),
        f"{prefix}/median": float(np.median(n)),
        f"{prefix}/max": float(n.max()),
        f"{prefix}/frac_zero": float((n == 0).mean()),
        # Sum of squared shares: 1/B when uniform, 1 when one prompt takes all.
        f"{prefix}/concentration": float((share**2).sum()),
        f"{prefix}/uniform_concentration": float(1.0 / len(n)),
    }
    if p is not None:
        p = np.asarray(p, dtype=float)
        out.update({
            f"{prefix}/p_mean": float(p.mean()),
            f"{prefix}/p_spread": float(p.std()),
            f"{prefix}/p_frac_zero": float((p <= 0).mean()),
            f"{prefix}/p_frac_one": float((p >= 1).mean()),
            # Prompts at p=0 or p=1 give zero advantage under mean-subtraction,
            # so they contribute no gradient no matter how many rollouts they get.
            f"{prefix}/p_frac_degenerate": float(((p <= 0) | (p >= 1)).mean()),
        })
    return out


def rollout_accounting(n_train, n_pilot=0, prefix="cost"):
    """Total rollouts actually spent, including any pilot pass.

    GVM's budget C covers only the training rollouts; the N' pilot rollouts per
    prompt are additional. Reporting `total` is what makes an equal-budget
    comparison against uniform or VIP honest.
    """
    train = float(np.sum(n_train))
    pilot = float(np.sum(n_pilot))
    return {
        f"{prefix}/rollouts_train": train,
        f"{prefix}/rollouts_pilot": pilot,
        f"{prefix}/rollouts_total": train + pilot,
        f"{prefix}/pilot_overhead": (pilot / train) if train > 0 else 0.0,
    }
