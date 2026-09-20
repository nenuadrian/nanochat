"""TPO invariants (arXiv:2604.06159). These run on CPU without a model."""
import pytest
import torch

from nanochat.tpo import standardize_scores, tpo_target, tpo_weights


def _group(seed, K=8):
    g = torch.Generator().manual_seed(seed)
    ell = torch.randn(K, generator=g) * 3.0          # sequence log-probs
    r = (torch.rand(K, generator=g) < 0.4).float()   # binary task reward
    if r.std(unbiased=False) == 0:                   # keep the group informative
        r[0] = 1.0 - r[0]
    return ell, r


@pytest.mark.parametrize("seed", range(10))
def test_surrogate_gradient_equals_cross_entropy_gradient(seed):
    """The whole reason chat_rl can microbatch a group: d/d ell of the real loss
    is exactly p^theta - q, so -(w * ell).sum() with w detached is equivalent."""
    ell, r = _group(seed)
    u = standardize_scores(r)
    q = tpo_target(ell, u).detach()

    a = ell.clone().requires_grad_(True)
    true_loss = -(q * torch.log_softmax(a, dim=-1)).sum()
    g_true, = torch.autograd.grad(true_loss, a)

    w, _, _ = tpo_weights(ell, r)
    b = ell.clone().requires_grad_(True)
    g_surr, = torch.autograd.grad(-(w.detach() * b).sum(), b)

    assert torch.allclose(g_true, g_surr, atol=1e-6)
    assert torch.allclose(g_true, torch.softmax(ell, -1) - q, atol=1e-6)


@pytest.mark.parametrize("seed", range(10))
def test_target_maximises_the_kl_regularised_objective(seed):
    """Proposition 1: q = argmax_r  E_r[u] - eta * KL(r || p_old) on the simplex."""
    ell, r = _group(seed)
    eta = 0.7
    u = standardize_scores(r)
    p_old = torch.softmax(ell, dim=-1)
    q = tpo_target(ell, u, eta=eta)

    def objective(x):
        x = x.clamp_min(1e-12)
        return float((x * u).sum() - eta * (x * (x / p_old).log()).sum())

    best = objective(q)
    g = torch.Generator().manual_seed(seed + 1000)
    for _ in range(200):
        # Dirichlet samples plus small perturbations of q itself, so the check
        # covers both the far field and the neighbourhood of the claimed optimum.
        cand = torch.distributions.Dirichlet(torch.ones_like(q)).sample()
        assert objective(cand) <= best + 1e-6
        eps = torch.randn(q.numel(), generator=g) * 0.01
        near = (q + eps - (q + eps).min().clamp(max=0.0)).clamp_min(1e-9)
        assert objective(near / near.sum()) <= best + 1e-6


@pytest.mark.parametrize("reward", [0.0, 1.0, 0.5])
def test_uniform_reward_group_contributes_nothing(reward):
    """All-fail and all-pass groups get u = 0, hence q = p_old and zero gradient.
    This is the free version of zero-variance masking -- the same prompts chat_rl
    already counts as realised/p_frac_degenerate."""
    ell = torch.randn(16) * 4.0
    w, q, diag = tpo_weights(ell, torch.full((16,), reward))
    assert torch.allclose(w, torch.zeros_like(w), atol=1e-6)
    assert torch.allclose(q, torch.softmax(ell, -1), atol=1e-6)
    assert diag["weight_l1"] < 1e-6


def test_single_candidate_group_is_safe():
    """n_i = 1 happens under the gvm/vip allocators; std of one sample is 0."""
    w, q, _ = tpo_weights(torch.tensor([-12.5]), torch.tensor([1.0]))
    assert torch.isfinite(w).all() and abs(float(w.sum())) < 1e-6


@pytest.mark.parametrize("seed", range(10))
def test_weights_sum_to_zero_like_a_mean_subtracted_advantage(seed):
    ell, r = _group(seed)
    w, q, _ = tpo_weights(ell, r)
    assert abs(float(w.sum())) < 1e-5
    assert abs(float(q.sum()) - 1.0) < 1e-5


@pytest.mark.parametrize("seed", range(10))
def test_target_moves_mass_toward_higher_reward(seed):
    """q/p_old must be monotone in the score: that is the whole point of the tilt."""
    ell, r = _group(seed)
    p = torch.softmax(ell, dim=-1)
    q = tpo_target(ell, standardize_scores(r))
    ratio = q / p
    hi, lo = ratio[r > r.mean()], ratio[r < r.mean()]
    assert hi.min() > lo.max()
    # and mass strictly moves, rather than the ranking merely being preserved
    assert float(q[r > r.mean()].sum()) > float(p[r > r.mean()].sum())


def test_standardize_uses_the_population_std():
    """Eq. 8 divides by K, not K-1; with K-1 the tilt would be systematically
    sharper on small groups, which is exactly where allocators put us."""
    s = torch.tensor([1.0, 0.0, 0.0, 0.0])
    u = standardize_scores(s)
    assert torch.allclose(u, (s - s.mean()) / s.std(unbiased=False), atol=1e-6)
    assert abs(float(u.max()) - 3.0 ** 0.5) < 1e-5   # z-score of one-hot over K=4


def test_temperature_limits():
    """Large eta -> no tilt at all; small eta -> all mass on the best candidate."""
    ell = torch.randn(8) * 2.0
    r = torch.tensor([0., 0., 1., 0., 1., 0., 0., 0.])
    u = standardize_scores(r)
    assert torch.allclose(tpo_target(ell, u, eta=1e6), torch.softmax(ell, -1), atol=1e-4)
    cold = tpo_target(ell, u, eta=1e-3)
    assert float(cold[r > 0].sum()) > 0.999


def test_anchor_ablation_drops_p_old():
    """--tpo-anchor 0: q ∝ exp(u), which the paper reports as consistently worse."""
    ell = torch.randn(8) * 5.0
    r = torch.tensor([0., 0., 1., 0., 1., 0., 0., 0.])
    u = standardize_scores(r)
    q = tpo_target(ell, u, anchor=False)
    assert torch.allclose(q, torch.softmax(u, -1), atol=1e-6)
    # every candidate with the same score gets the same mass once p_old is gone
    assert abs(float(q[r == 0].std(unbiased=False))) < 1e-6


def test_frozen_target_is_reused_across_gradient_epochs():
    """Multi-epoch reuse keeps q fixed and only re-measures p^theta (paper 2)."""
    ell_old, r = _group(3)
    _, q0, _ = tpo_weights(ell_old, r)
    ell_new = ell_old + torch.randn_like(ell_old)     # policy has since moved
    w, q1, _ = tpo_weights(ell_new, r, target=q0)
    assert torch.allclose(q0, q1)
    assert torch.allclose(w, q0 - torch.softmax(ell_new, -1), atol=1e-6)


def test_collapsed_group_is_visible_in_the_diagnostics():
    """Long completions can drive the group softmax to one-hot, at which point
    TPO has nothing left to redistribute. chat_rl logs p_max so a run can be
    checked for this rather than assumed healthy."""
    ell = torch.tensor([-40.0, -200.0, -260.0, -310.0])   # wildly different lengths
    r = torch.tensor([0.0, 1.0, 1.0, 0.0])
    w, _, diag = tpo_weights(ell, r)
    assert diag["p_max"] > 0.999
    assert diag["weight_l1"] < 1e-3          # reward ranking cannot overcome it
    # length-normalising the same group restores a usable signal
    tok = torch.tensor([20.0, 100.0, 130.0, 155.0])
    w_norm, _, diag_norm = tpo_weights(ell / tok, r)
    assert diag_norm["weight_l1"] > 0.5
