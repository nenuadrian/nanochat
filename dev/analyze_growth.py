"""
Paired analysis for the capacity-growth RL experiment (runs/growth_rl.sh).

Why paired: essentially all of the step-to-step reward noise in this setup is
which prompts happened to land in that step -- measured on a real 140-step run,
prompt-difficulty variance accounts for ~100% of the detrended step-to-step
variance. Two arms run at the same --seed see the *same* prompts in the same
order with the same rollout seeds, so that noise is common and cancels in a
difference. Comparing arm means independently throws that away and needs roughly
an order of magnitude more runs to see the same effect.

Usage:  python dev/analyze_growth.py [logdir]        (default: logs/)
"""

import os
import re
import sys
import collections
import numpy as np

STEP_RE = re.compile(r"^Step (\d+)/(\d+) \| Average reward: ([0-9.]+) .*?\| ([0-9.]+)s\s*$")
STEP_RE_NOTIME = re.compile(r"^Step (\d+)/(\d+) \| Average reward: ([0-9.]+)")
GROW_RE = re.compile(r"^\[grow\] step (\d+): (\d+) -> (\d+) layers")
BUMP_RE = re.compile(r"^\[lr-bump\] step (\d+):")
DIAG_RE = re.compile(
    r"^Step (\d+) \| diag \| entropy ([0-9.]+) \| dormant ([0-9.]+) \| eff_rank ([0-9.]+)"
    r" \| c_proj spec old ([0-9.]+)(?: new ([0-9.]+))?")
NAME_RE = re.compile(r"growth-(.+)-s(\d+)\.log$")


def parse(path):
    r = {"reward": {}, "secs": {}, "diag": {}, "grow_steps": [], "bump_steps": []}
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = STEP_RE.match(line) or STEP_RE_NOTIME.match(line)
            if m:
                r["reward"][int(m.group(1))] = float(m.group(3))
                if m.re is STEP_RE:
                    r["secs"][int(m.group(1))] = float(m.group(4))
                continue
            m = DIAG_RE.match(line)
            if m:
                r["diag"][int(m.group(1))] = dict(
                    entropy=float(m.group(2)), dormant=float(m.group(3)),
                    eff_rank=float(m.group(4)), spec_old=float(m.group(5)),
                    spec_new=float(m.group(6)) if m.group(6) else 0.0)
                continue
            m = GROW_RE.match(line)
            if m:
                r["grow_steps"].append(int(m.group(1)))
                continue
            m = BUMP_RE.match(line)
            if m:
                r["bump_steps"].append(int(m.group(1)))
    return r


def bootstrap_ci(x, n=10000, alpha=0.05, seed=0):
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def main(logdir="logs"):
    runs = {}
    for fn in sorted(os.listdir(logdir)):
        m = NAME_RE.search(fn)
        if m:
            runs[(m.group(1), int(m.group(2)))] = parse(os.path.join(logdir, fn))
    if not runs:
        print(f"no growth-*-s*.log files in {logdir}/")
        return
    arms = sorted({a for a, _ in runs})
    seeds = sorted({s for _, s in runs})
    print(f"{len(runs)} runs | arms: {', '.join(arms)} | seeds: {seeds}\n")

    if "static" not in arms:
        print("no 'static' arm: nothing to pair against.")
        return

    # Intervention step, taken from the arms that have one.
    interventions = [s for r in runs.values() for s in r["grow_steps"] + r["bump_steps"]]
    g = min(interventions) if interventions else None
    print(f"intervention step: {g}\n" if g is not None else "no intervention found\n")

    # --- 1. pairing check: identical seeds must agree bit-for-bit before the seam.
    # With identity-preserving growth the policy is unchanged at step g itself, so
    # the rollouts at g are identical too and divergence starts at g+1.
    if g is not None:
        print("[1] pairing check (arms at the same seed must match exactly up to step g)")
        for arm in arms:
            if arm == "static":
                continue
            bad = []
            for s in seeds:
                a, b = runs.get((arm, s)), runs.get(("static", s))
                if not a or not b:
                    continue
                shared = [k for k in a["reward"] if k in b["reward"] and k <= g]
                d = max((abs(a["reward"][k] - b["reward"][k]) for k in shared), default=0.0)
                if d > 1e-9:
                    bad.append(f"s{s}:{d:.2e}")
            print(f"    {arm:<14} {'OK' if not bad else 'DIVERGED before the seam -> ' + ','.join(bad)}")
        print()

    # --- 2. the actual result: paired difference vs static after the intervention
    print("[2] paired difference in mean reward vs static (post-intervention)")
    print(f"    {'arm':<14} {'window':<12} {'diff':>8} {'95% CI':>20} {'per-seed':>26}")
    last = max(max(r["reward"]) for r in runs.values() if r["reward"])
    windows = [(g + 1, last)] if g is not None else []
    if g is not None and last - g > 60:
        windows.insert(0, (g + 1, g + 30))          # the transient
        windows.append((last - 40, last))            # the endgame
    for arm in arms:
        if arm == "static":
            continue
        for lo, hi in windows:
            per_seed, pooled = [], []
            for s in seeds:
                a, b = runs.get((arm, s)), runs.get(("static", s))
                if not a or not b:
                    continue
                ks = [k for k in a["reward"] if k in b["reward"] and lo <= k <= hi]
                if not ks:
                    continue
                d = np.array([a["reward"][k] - b["reward"][k] for k in ks])
                per_seed.append(d.mean())
                pooled.extend(d)
            if not pooled:
                continue
            pooled = np.array(pooled)
            ci = bootstrap_ci(pooled)
            verdict = "  <- excludes 0" if (ci[0] > 0 or ci[1] < 0) else ""
            print(f"    {arm:<14} {f'{lo}-{hi}':<12} {pooled.mean():>+8.4f} "
                  f"{f'[{ci[0]:+.4f}, {ci[1]:+.4f}]':>20} "
                  f"{' '.join(f'{v:+.3f}' for v in per_seed):>26}{verdict}")
    print("\n    Note: per-seed columns are the paired difference for that seed. If they do not")
    print("    agree in sign, the pooled CI is optimistic -- seed-level spread is the real error bar.\n")

    # --- 3. did the grown capacity ever engage?
    print("[3] did the grown layers engage? (mean |c_proj| spectral norm, new vs old)")
    for arm in arms:
        rows = [runs[(arm, s)] for s in seeds if (arm, s) in runs and runs[(arm, s)]["diag"]]
        if not rows:
            continue
        finals = [r["diag"][max(r["diag"])] for r in rows]
        new = np.mean([f["spec_new"] for f in finals])
        old = np.mean([f["spec_old"] for f in finals])
        if new == 0:
            print(f"    {arm:<14} no grown layers")
        else:
            flag = "   <- NO-OP: this arm says nothing about capacity" if new < 0.1 * old else ""
            print(f"    {arm:<14} new {new:>6.3f}  vs old {old:>6.3f}  ({100*new/old:.0f}%){flag}")
    print()

    # --- 4. was there any plasticity loss to restore?
    print("[4] plasticity: is anything actually degrading over the run?")
    print(f"    {'arm':<14} {'entropy':>18} {'dormant':>18} {'eff_rank':>18}")
    for arm in arms:
        rows = [runs[(arm, s)] for s in seeds if (arm, s) in runs and len(runs[(arm, s)]["diag"]) > 1]
        if not rows:
            continue
        cells = []
        for key in ("entropy", "dormant", "eff_rank"):   # entropy = decision-position
            a = np.mean([r["diag"][min(r["diag"])][key] for r in rows])
            b = np.mean([r["diag"][max(r["diag"])][key] for r in rows])
            cells.append(f"{a:.3f} -> {b:.3f}")
        print(f"    {arm:<14} {cells[0]:>18} {cells[1]:>18} {cells[2]:>18}")
    print("\n    If entropy does not fall and dormant units do not accumulate in the static arm,")
    print("    there is no plasticity loss in this regime -- and growth has nothing to restore.")
    print("    That is a real finding, and it reframes the experiment as a pure capacity test.\n")

    # --- 5. compute, so the comparison is not silently steps-matched
    if any(r["secs"] for r in runs.values()):
        print("[5] cost (train seconds/step, excluding evaluation)")
        for arm in arms:
            vals = [v for s in seeds if (arm, s) in runs for v in runs[(arm, s)]["secs"].values()]
            if vals:
                print(f"    {arm:<14} {np.mean(vals):.2f}s/step   total {np.sum(vals)/3600:.2f} h over {len(seeds)} seeds")
        print("\n    Growth raises the per-step cost. An equal-STEPS win that disappears at equal")
        print("    wall-clock is not a win; report both.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs")
