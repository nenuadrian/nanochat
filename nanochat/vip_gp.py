"""Gaussian process over prompt embeddings, predicting per-prompt success rate.

This is VIP's substitute for GVM's pilot rollouts (arXiv:2602.01601 section 5.1).
Where GVM spends N' extra rollouts per prompt to MEASURE p_q, VIP predicts it
from the prompt embedding and refines the belief from rollouts it was going to
draw anyway -- so the prediction is free, at the cost of being wrong early on.

Model, following the paper:

    p_{q,t} = sigmoid(g_t(x_q)),   g_t ~ GP(m_t, K),   K = RBF(bandwidth h)

    m_1 = 0                                            (zero-mean prior)
    after a batch B with mean reward Rbar_q in [-1, 1]:
        phat_q = clip((Rbar_q + 1) / 2, eps, 1 - eps)
        ghat_q = logit(phat_q)
        m_{t+1}[B]   = ghat_B                          (observed)
        m_{t+1}[B^c] = m[B^c] + K_{B^c,B} K_{B,B}^-1 (ghat_B - m[B])

The full Q x Q kernel is never formed. Only the Q x |B| and |B| x |B| blocks are
needed, and |B| is the minibatch size, so each update is cheap even for a large
prompt pool.
"""

import numpy as np

__all__ = ["PromptSuccessGP"]


class PromptSuccessGP:
    def __init__(self, embeddings, bandwidth=None, eps=0.05, jitter=1e-4, reward_range=(0.0, 1.0)):
        """embeddings: (Q, d) array, one row per training prompt."""
        X = np.asarray(embeddings, dtype=np.float64)
        # Unit-normalise so the RBF bandwidth means the same thing regardless of
        # the model's embedding scale, which drifts a lot during training.
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        self.X = X / np.maximum(norms, 1e-9)
        self.Q = len(X)
        self.eps = float(eps)
        self.jitter = float(jitter)
        self.r_lo, self.r_hi = reward_range
        if bandwidth is None:
            # Median heuristic on a subsample: a bandwidth that makes typical
            # pairs neither perfectly correlated nor independent.
            idx = np.random.default_rng(0).choice(self.Q, size=min(512, self.Q), replace=False)
            d2 = np.sum((self.X[idx][:, None, :] - self.X[idx][None, :, :]) ** 2, axis=-1)
            med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
            bandwidth = float(np.sqrt(max(med, 1e-9) / 2.0))
        self.h = float(bandwidth)
        self.m = np.zeros(self.Q, dtype=np.float64)   # prior mean over latent g
        self.n_updates = 0

    def _k(self, A, B):
        d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return np.exp(-d2 / (2.0 * self.h**2))

    def predict(self, idx):
        """Predicted success probability for prompt indices `idx`."""
        idx = np.asarray(idx, dtype=int)
        return 1.0 / (1.0 + np.exp(-self.m[idx]))

    def _reward_to_p(self, mean_reward):
        """Map a mean reward onto (0,1), clipped away from the logit's poles."""
        span = max(self.r_hi - self.r_lo, 1e-9)
        p = (np.asarray(mean_reward, dtype=np.float64) - self.r_lo) / span
        return np.clip(p, self.eps, 1.0 - self.eps)

    def update(self, idx, mean_reward):
        """Recursive posterior update from the batch's realised mean rewards."""
        idx = np.asarray(idx, dtype=int)
        p_obs = self._reward_to_p(mean_reward)
        g_obs = np.log(p_obs / (1.0 - p_obs))

        Xb = self.X[idx]
        Kbb = self._k(Xb, Xb) + self.jitter * np.eye(len(idx))
        resid = g_obs - self.m[idx]
        try:
            alpha = np.linalg.solve(Kbb, resid)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(Kbb, resid, rcond=None)[0]
        if not np.all(np.isfinite(alpha)):
            # Near-duplicate prompts make Kbb singular. Fall back to the
            # least-squares solution rather than poisoning every future
            # allocation with inf, which would fail silently downstream.
            alpha = np.linalg.lstsq(Kbb, resid, rcond=None)[0]
            alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

        # Propagate to every prompt, then overwrite the observed ones exactly.
        # Apple's Accelerate BLAS raises spurious FP warnings from matmul even
        # when every value is finite; the explicit nan_to_num and clip below are
        # what actually guarantee the result, so the warning is just noise here.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            self.m = self.m + self._k(self.X, Xb) @ alpha
        self.m[idx] = g_obs
        # The latent is a logit of a probability clipped to [eps, 1-eps], so it
        # is bounded by construction; clipping keeps extrapolation to unqueried
        # prompts from drifting into overflow over many updates.
        bound = abs(np.log((1 - self.eps) / self.eps)) * 2.0
        self.m = np.clip(np.nan_to_num(self.m, nan=0.0), -bound, bound)
        self.n_updates += 1

    def calibration(self, idx, realised_mean_reward, prefix="vip_gp"):
        """How wrong was the prediction? The metric that says whether VIP's
        free p_q is actually usable, or whether the GP is just noise."""
        idx = np.asarray(idx, dtype=int)
        pred = self.predict(idx)
        real = self._reward_to_p(realised_mean_reward)
        err = pred - real
        out = {
            f"{prefix}/pred_mean": float(pred.mean()),
            f"{prefix}/real_mean": float(real.mean()),
            f"{prefix}/mae": float(np.abs(err).mean()),
            f"{prefix}/bias": float(err.mean()),
            f"{prefix}/bandwidth": self.h,
            f"{prefix}/updates": float(self.n_updates),
        }
        if len(idx) > 2 and pred.std() > 1e-9 and real.std() > 1e-9:
            out[f"{prefix}/corr"] = float(np.corrcoef(pred, real)[0, 1])
        else:
            # No spread means the correlation is undefined, not zero -- worth
            # distinguishing, since a flat prediction is exactly the failure mode.
            out[f"{prefix}/corr"] = float("nan")
        return out
