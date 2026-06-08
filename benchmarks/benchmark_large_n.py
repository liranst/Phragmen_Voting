"""PP-Phragmén large-N runtime experiment with linear extrapolation.

Runs a single election period (R=1) over N in [500, 1000, 2000, 5000]
using the small field p=251 and M=3 candidates for speed.  Fits a linear
model t(N) = a·N + b and extrapolates to N = 10^5.

Reference: "Fairness Without Exposure: Privacy-Preserving Phragmén Voting."
"""

import statistics
import sys
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from protocol.run_election import run_election  # noqa: E402

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
M: int = 3
P: int = 251
N_PARTIES: int = 3
THRESHOLD: int = 2
R: int = 1
REPS: int = 3

N_SWEEP: List[int] = [500, 1000, 2000, 5000]
EXTRAPOLATE_N: int = 100_000

OUTPUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ballots(n_voters: int, n_candidates: int) -> List[List[int]]:
    """Return an N×M ballot matrix where voter i approves candidate (i mod M)."""
    row_templates = []
    for c in range(n_candidates):
        row = [0] * n_candidates
        row[c] = 1
        row_templates.append(row)
    return [row_templates[i % n_candidates] for i in range(n_voters)]


def time_once(ballots: List[List[int]], seed: int = 0) -> float:
    """Return wall-clock seconds for run_election() with R=1."""
    t0 = time.perf_counter()
    run_election(
        [ballots],
        p=P,
        n_parties=N_PARTIES,
        threshold=THRESHOLD,
        seed=seed,
    )
    return time.perf_counter() - t0


def measure(n_voters: int) -> Tuple[float, float]:
    """Return (mean_s, std_s) over REPS timing runs for the given N."""
    ballots = make_ballots(n_voters, M)
    times = [time_once(ballots, seed=k) for k in range(REPS)]
    return statistics.mean(times), statistics.pstdev(times) if REPS > 1 else 0.0


# ---------------------------------------------------------------------------
# Linear fit and extrapolation
# ---------------------------------------------------------------------------

def fit_linear(Ns: List[int], means: List[float]) -> Tuple[float, float]:
    """Fit t = a·N + b; return (a, b)."""
    coeffs = np.polyfit(Ns, means, deg=1)
    return float(coeffs[0]), float(coeffs[1])


def r_squared(Ns: List[int], means: List[float], a: float, b: float) -> float:
    """Coefficient of determination for the linear fit."""
    y = np.array(means)
    y_hat = a * np.array(Ns) + b
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("PP-Phragmén Large-N Experiment")
    print(f"  M={M}, p={P}, R={R}, parties={N_PARTIES}, threshold={THRESHOLD}, reps={REPS}\n")

    Ns: List[int] = []
    means: List[float] = []
    stds: List[float] = []

    print("Collecting timings...")
    for N in N_SWEEP:
        mean, std = measure(N)
        Ns.append(N)
        means.append(mean)
        stds.append(std)
        print(f"  N={N:>5}: {mean:.4f} s  ± {std:.4f}")

    # Linear fit
    a, b = fit_linear(Ns, means)
    r2 = r_squared(Ns, means, a, b)
    extrap = a * EXTRAPOLATE_N + b

    # ---------------------------------------------------------------------------
    # Print table
    # ---------------------------------------------------------------------------
    print()
    print("=" * 62)
    print(f"Large-N Experiment (M={M}, R={R}, p={P})")
    print("=" * 62)
    print(f"{'N':>7} | {'mean (s)':>10} | {'std (s)':>9} | {'fit (s)':>9} | {'residual':>10}")
    print("-" * 7 + "+-" + "-" * 10 + "+-" + "-" * 9 + "+-" + "-" * 9 + "+-" + "-" * 10)
    for N, mean, std in zip(Ns, means, stds):
        fit_val = a * N + b
        resid = mean - fit_val
        print(f"{N:>7} | {mean:>10.4f} | {std:>9.4f} | {fit_val:>9.4f} | {resid:>+10.4f}")

    print()
    print(f"Linear fit:   t(N) = {a:.6e} * N + {b:.4f}")
    print(f"R^2         = {r2:.6f}")
    print()
    print("=" * 40)
    print(f"Extrapolation to N = {EXTRAPOLATE_N:,}")
    print("=" * 40)
    print(f"  t(N=10^5) ~ {extrap:.2f} s  ({extrap/60:.2f} min)")
    print(f"  = {a:.6e} * {EXTRAPOLATE_N} + {b:.4f}")

    # Confidence interval: 2x largest residual, scaled to extrapolation distance
    residuals = [abs(means[i] - (a * Ns[i] + b)) for i in range(len(Ns))]
    margin = 2.0 * max(residuals) * (EXTRAPOLATE_N / max(Ns))
    print(f"  Conservative bound: +/-{margin:.1f} s")

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    N_dense = np.linspace(min(Ns), EXTRAPOLATE_N, 500)
    fit_line = a * N_dense + b

    fig, ax = plt.subplots(figsize=(8, 5))

    # Measured points with error bars
    ax.errorbar(Ns, means, yerr=stds, fmt="o", color="steelblue",
                markersize=7, capsize=5, linewidth=1.5, label="Measured (mean ± std)")

    # Linear fit line extended to 10^5
    ax.plot(N_dense, fit_line, "--", color="tomato", linewidth=2,
            label=f"Linear fit: t={a:.2e}·N+{b:.3f}  (R²={r2:.4f})")

    # Extrapolation marker
    ax.scatter([EXTRAPOLATE_N], [extrap], marker="*", s=200, color="darkorange",
               zorder=5, label=f"Extrapolated N=10⁵: {extrap:.1f} s")
    ax.axvline(EXTRAPOLATE_N, color="darkorange", linestyle=":", linewidth=1, alpha=0.6)

    ax.set_xlabel("Number of Voters (N)", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title(f"PP-Phragmén Runtime vs N (M={M}, R={R}, p={P})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Secondary x-axis label for the extrapolation tick
    ax.set_xticks(list(Ns) + [EXTRAPOLATE_N])
    ax.set_xticklabels([str(n) for n in Ns] + ["10⁵"], rotation=30, ha="right", fontsize=9)

    path = OUTPUT_DIR / "plot_large_n_extrapolate.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"\nPlot saved → {path}")


if __name__ == "__main__":
    main()
