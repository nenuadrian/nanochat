"""Target Policy Optimization (TPO), arXiv:2604.06159.

GRPO answers two questions at once: which completions should gain probability
mass, and how far the parameters should move to make that happen. TPO separates
them. Given a group of K scored completions it first writes down the
distribution it *wants* over those K candidates,

    q_i  =  p_i^old * exp(u_i / eta)  /  Z,

where p^old is the policy's own (renormalised) distribution over the group at
rollout time and u is the within-group z-score of the task reward, and then fits
the policy to q by cross-entropy:

    L_TPO = - sum_i q_i log p_i^theta,          p^theta = softmax(ell^theta).

q is the closed-form argmax of  E_q[u] - eta * KL(q || p^old)  over the simplex
on the sampled candidates (Prop. 1), so TPO is a KL-regularised improvement step
restricted to the K completions we actually drew -- MPO's E-step without needing
MPO's critic.

Two properties matter for us:

  * dL/d ell_i = p_i^theta - q_i, so the gradient vanishes exactly when the
    policy matches the target. A mean-subtracted policy gradient has no such
    fixed point: it keeps pushing on a group it has already solved.
  * a group where every completion scores the same has sigma = 0, hence u = 0,
    hence q = p^old and zero gradient -- for free, with no zero-variance
    masking. That is the same set of prompts nanochat currently logs as
    `realised/p_frac_degenerate`.

THE SURROGATE. p^theta couples all K candidates through one softmax, so a naive
implementation needs the whole group live in one autograd graph. We do not: the
gradient wrt the sequence log-prob is exactly (p_i^theta - q_i), so

    L_surrogate = - sum_i w_i * ell_i^theta,     w = (q - p^theta).detach()

has the *same* gradient as L_TPO at the current theta. That is the identical
`(coefficient) x (log-prob)` shape the existing REINFORCE path already uses, so
TPO drops into chat_rl's microbatched loop with w in place of the advantage, and
groups may be split across as many forward passes as memory requires. Note the
coefficients are only exact for the theta that produced ell^theta; reusing a
rollout batch for more gradient epochs means recomputing p^theta (q stays
frozen), which is what `tpo_weights` is called again for.

CAVEAT worth measuring before trusting any result. ell_i is a *sum* of token
log-probs, so for long completions the group softmax is dominated by whichever
candidate happened to be shortest/likeliest: p^theta goes near one-hot, q ~ p^old
however the rewards fall, and w -> 0. Short-completion tasks (ARC emits a letter)
are safe; 256-token GSM8K chains are not obviously safe. `tpo_weights` therefore
returns p_max and the total weight mass so a run can be checked rather than
assumed, and chat_rl exposes --tpo-logp=mean to divide ell by its token count
(cheap fix, GSPO's reasoning, but no longer the paper's objective).
"""

import torch

__all__ = ["standardize_scores", "tpo_target", "tpo_weights"]


def standardize_scores(scores):
    """Within-group z-score of the raw task scores (paper Eq. 8).

    Uses the POPULATION standard deviation (divide by K, not K-1) and maps the
    zero-variance group to all-zeros, which is what makes an all-fail or
    all-pass group contribute exactly nothing.

    Standardisation is what lets eta=1 be a sane default: the target
    exponentiates u, so without it a scorer that returns (100, 0, -100) and one
    that returns (1, 0, -1) -- same ranking -- would produce wildly different
    targets.
    """
    s = scores.float()
    sigma = s.std(unbiased=False)
    if not torch.isfinite(sigma) or sigma <= 0:
        return torch.zeros_like(s)
    return (s - s.mean()) / sigma


def tpo_target(seq_logprobs, u, eta=1.0, anchor=True):
    """q_i proportional to p_i^old * exp(u_i / eta), over one group.

    `seq_logprobs` are the un-normalised sequence log-probs ell^old; the
    log_softmax below is what turns them into p^old over the group. With
    anchor=False the p^old term is dropped (q ∝ exp(u/eta)), the ablation the
    paper reports as consistently harmful -- kept here because it is the one
    knob that tells you whether the anchor is doing the work.
    """
    log_q = u / eta
    if anchor:
        log_q = torch.log_softmax(seq_logprobs.float(), dim=-1) + log_q
    return torch.softmax(log_q, dim=-1)


def tpo_weights(seq_logprobs, scores, eta=1.0, anchor=True, target=None):
    """Per-candidate surrogate coefficients w = q - p^theta for one group.

    Args:
        seq_logprobs: (K,) sequence log-probs of the sampled completions under
            the CURRENT policy. On the first gradient epoch these are also
            ell^old, which is why the on-policy update needs a single forward.
        scores: (K,) raw task rewards.
        eta: target temperature; 1.0 throughout the paper and robust over
            roughly [0.25, 2].
        anchor: keep the p^old anchor in the target.
        target: a q computed earlier (frozen across gradient epochs). When None,
            q is built from `seq_logprobs`, i.e. p^old = p^theta.

    Returns (w, q, diagnostics). w sums to zero, exactly like a mean-subtracted
    advantage, since q and p^theta are both distributions.
    """
    ell = seq_logprobs.float()
    u = standardize_scores(scores)
    q = tpo_target(ell, u, eta=eta, anchor=anchor) if target is None else target
    p = torch.softmax(ell, dim=-1)
    w = q - p
    diagnostics = {
        # p_max near 1 means the group softmax collapsed onto one completion and
        # TPO has almost no room left to redistribute -- see CAVEAT above.
        "p_max": float(p.max().item()),
        # Normalised entropy of p^theta, 1.0 = uniform over the group, 0 = collapsed.
        "p_entropy": float(
            (-(p * p.clamp_min(1e-12).log()).sum() / torch.log(torch.tensor(float(max(p.numel(), 2))))).item()
        ),
        # sum |q - p| = 2 * total variation, in [0, 2]. This is the size of the
        # redistribution TPO asked for; 0 means the step was a no-op.
        "weight_l1": float(w.abs().sum().item()),
        "u_absmax": float(u.abs().max().item()),
    }
    return w, q, diagnostics
