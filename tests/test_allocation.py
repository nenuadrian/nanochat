"""Allocator invariants. These run without a model or any GPU."""
import numpy as np
import pytest

from nanochat.allocation import (
    uniform_allocation, gvm_allocation, vip_allocation,
    round_preserving_sum, allocation_stats, rollout_accounting,
)
from nanochat.vip_gp import PromptSuccessGP


@pytest.mark.parametrize("seed", range(20))
def test_budget_is_conserved_exactly(seed):
    rng = np.random.default_rng(seed)
    B = int(rng.integers(4, 32))
    C = int(B * rng.integers(4, 16))
    p = rng.random(B)
    p[rng.random(B) < 0.3] = 0.0
    G = rng.random(B) * 100
    assert uniform_allocation(B, C).sum() == C
    assert gvm_allocation(p, G, C).sum() == C
    assert vip_allocation(np.clip(p, 0.01, 0.99), C, lo=3, hi=C).sum() == C


def test_bounds_are_respected():
    p = np.clip(np.random.default_rng(0).random(16), 0.01, 0.99)
    n = vip_allocation(p, 16 * 10, lo=3, hi=20)
    assert n.min() >= 3 and n.max() <= 20 and n.sum() == 160


def test_vip_beats_uniform_on_its_own_objective():
    """VIP minimises sum_q a_q (n_q-1)/n_q^2; it should not lose to uniform."""
    def var(n, p):
        n = np.maximum(np.asarray(n, float), 1e-9)
        return float(np.sum(4 * p * (1 - p) * (n - 1) / n ** 2))
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.random(16) * 0.9 + 0.05
        C = 16 * 12
        assert var(vip_allocation(p, C, lo=3, hi=48), p) <= var(uniform_allocation(16, C, lo=3), p) + 1e-9


def test_vip_peaks_at_half_and_gvm_peaks_at_hard():
    """The two allocators genuinely disagree -- that is why both are worth running."""
    p = np.array([0.02, 0.25, 0.50, 0.75, 0.98])
    G = np.ones(5)
    assert np.argmax(vip_allocation(p, 5 * 16, lo=3, hi=60)) == 2      # p = 0.5
    assert np.argmax(gvm_allocation(p, G, 5 * 16)) == 0                # lowest p


def test_gvm_zeroes_unsolved_prompts():
    p = np.array([0.0, 0.0, 0.5, 0.25])
    G = np.array([0.0, 0.0, 10.0, 8.0])
    n = gvm_allocation(p, G, 40)
    assert n[0] == 0 and n[1] == 0 and n.sum() == 40


def test_gvm_falls_back_to_uniform_when_nothing_is_solvable():
    """All-zero p must not produce a divide-by-zero or an empty batch."""
    n = gvm_allocation(np.zeros(8), np.zeros(8), 64)
    assert n.sum() == 64


def test_round_preserving_sum_respects_bounds():
    x = np.array([0.1, 0.2, 9.7])
    n = round_preserving_sum(x * 10, 100, lo=3, hi=50)
    assert n.sum() == 100 and n.min() >= 3 and n.max() <= 50


def test_gp_predicts_unqueried_prompts():
    """VIP's whole premise: p_q for prompts it never sampled."""
    rng = np.random.default_rng(0)
    Q, d = 300, 16
    X = rng.normal(size=(Q, d))
    w = rng.normal(size=d)
    p_true = 1 / (1 + np.exp(-np.clip(X @ w / np.sqrt(d), -30, 30)))
    gp = PromptSuccessGP(X, reward_range=(0.0, 1.0))
    for _ in range(40):
        idx = rng.choice(Q, size=16, replace=False)
        gp.update(idx, rng.binomial(8, p_true[idx]) / 8.0)
    held = np.setdiff1d(np.arange(Q), idx)[:150]
    assert np.all(np.isfinite(gp.m))
    assert np.corrcoef(gp.predict(held), p_true[held])[0, 1] > 0.4


def test_metrics_flag_a_degenerate_batch():
    """p=0 or p=1 give zero advantage under mean subtraction, so no allocator
    can help; the metric has to make that visible."""
    s = allocation_stats(uniform_allocation(5, 40), np.array([0., 0., 1., 1., .5]))
    assert s["alloc/p_frac_degenerate"] == pytest.approx(0.8)
    c = rollout_accounting(n_train=[8] * 16, n_pilot=[4] * 16)
    assert c["cost/rollouts_total"] == 192 and c["cost/pilot_overhead"] == pytest.approx(0.5)


# --- ARC reward parsing ------------------------------------------------------
# ARC.evaluate() asserts a bare letter, which only holds for the categorical eval
# harness. RL generates free text, so reward() must parse it and never raise.

def test_arc_extract_choice():
    from tasks.arc import ARC
    L = ["A", "B", "C", "D"]
    assert ARC.extract_choice("A", L) == "A"
    assert ARC.extract_choice("  B  ", L) == "B"
    assert ARC.extract_choice("C.", L) == "C"
    assert ARC.extract_choice("(D)", L) == "D"
    assert ARC.extract_choice("Answer: B", L) == "B"
    assert ARC.extract_choice("The answer is C.", L) == "C"
    # must not match a letter embedded in a word
    assert ARC.extract_choice("Apple", L) is None
    # unparseable is a legitimate wrong answer, not an error
    assert ARC.extract_choice("", L) is None
    assert ARC.extract_choice("banana", L) is None
    assert ARC.extract_choice("E", L) is None


def test_arc_reward_never_raises_on_garbage():
    from tasks.arc import ARC
    conv = {"messages": [{"role": "user", "content": "q"},
                         {"role": "assistant", "content": "B"}],
            "letters": ["A", "B", "C", "D"]}
    # Subclass to skip ARC.__init__, which would download the dataset.
    class _NoLoad(ARC):
        def __init__(self):
            pass
    task = _NoLoad()
    assert task.reward(conv, "B") == 1.0
    assert task.reward(conv, "Answer: B") == 1.0
    assert task.reward(conv, "A") == 0.0
    assert task.reward(conv, "complete nonsense") == 0.0
    assert task.reward(conv, "") == 0.0
