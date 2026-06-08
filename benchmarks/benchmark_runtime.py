"""PP-Phragmén runtime benchmarks.

Reproduces the spirit of Figures 1–4 from the SoK paper:
  Experiment 1 — vary N (voters),   fixed R=3
  Experiment 2 — vary R (rounds),   fixed N=20
  Experiment 3 — full N × R grid

Reference: "Fairness Without Exposure: Privacy-Preserving Phragmén Voting."
"""

import csv
import statistics
import sys
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Make the project root importable regardless of the working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from protocol.run_election import run_election  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------
M: int = 10                  # candidates
P: int = 2**61 - 1           # Mersenne prime, 61 bits
N_PARTIES: int = 3
THRESHOLD: int = 2
REPS: int = 3                # timing repetitions per configuration
P_BITS: int = 61

# Sweep values
N_SWEEP: List[int] = [5, 10, 20, 50, 100, 200]
R_SWEEP: List[int] = [1, 2, 3, 4, 5, 6]
N_GRID: List[int] = [5, 10, 20, 50]
R_GRID: List[int] = [1, 2, 3, 4, 5]

FIXED_R: int = 3
FIXED_N: int = 20

OUTPUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ballots(n_voters: int, n_candidates: int) -> List[List[int]]:
    """Return an N×M ballot matrix where voter i approves candidate (i mod M)."""
    ballots = []
    for i in range(n_voters):
        row = [0] * n_candidates
        row[i % n_candidates] = 1
        ballots.append(row)
    return ballots


def time_run_election(
    ballot_periods: List[List[List[int]]],
    seed: int = 0,
) -> float:
    """Return wall-clock seconds for a single run_election() call."""
    t0 = time.perf_counter()
    run_election(
        ballot_periods,
        p=P,
        n_parties=N_PARTIES,
        threshold=THRESHOLD,
        seed=seed,
    )
    return time.perf_counter() - t0


def measure(
    ballot_periods: List[List[List[int]]],
    reps: int = REPS,
) -> Tuple[float, float]:
    """Return (mean_seconds, std_seconds) over *reps* independent runs."""
    times = [time_run_election(ballot_periods, seed=k) for k in range(reps)]
    mean = statistics.mean(times)
    std = statistics.pstdev(times) if reps > 1 else 0.0
    return mean, std


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_experiment_1() -> List[dict]:
    """Vary N voters, fixed R=3."""
    print("\n" + "=" * 50)
    print(f"Experiment 1: Runtime vs N (M={M}, R={FIXED_R})")
    print("=" * 50)
    print(f"{'N':<6} | {'mean (s)':>10} | {'std (s)':>9} | {'ratio to N=5':>13}")
    print("-" * 6 + "+-" + "-" * 10 + "+-" + "-" * 9 + "+-" + "-" * 13)

    rows = []
    baseline_mean: float = 0.0

    for N in N_SWEEP:
        ballots = make_ballots(N, M)
        ballot_periods = [ballots] * FIXED_R
        mean, std = measure(ballot_periods)

        if N == N_SWEEP[0]:
            baseline_mean = mean

        ratio = mean / baseline_mean if baseline_mean > 0 else 1.0
        print(f"{N:<6} | {mean:>10.4f} | {std:>9.4f} | {ratio:>12.2f}x")

        rows.append(dict(
            experiment=1, N=N, M=M, R=FIXED_R,
            p_bits=P_BITS, mean_s=mean, std_s=std,
        ))

    return rows


def run_experiment_2() -> List[dict]:
    """Vary R rounds, fixed N=20."""
    print("\n" + "=" * 50)
    print(f"Experiment 2: Runtime vs R (M={M}, N={FIXED_N})")
    print("=" * 50)
    print(f"{'R':<6} | {'mean (s)':>10} | {'std (s)':>9} | {'ratio to R=1':>13}")
    print("-" * 6 + "+-" + "-" * 10 + "+-" + "-" * 9 + "+-" + "-" * 13)

    ballots = make_ballots(FIXED_N, M)
    rows = []
    baseline_mean: float = 0.0

    for R in R_SWEEP:
        ballot_periods = [ballots] * R
        mean, std = measure(ballot_periods)

        if R == R_SWEEP[0]:
            baseline_mean = mean

        ratio = mean / baseline_mean if baseline_mean > 0 else 1.0
        print(f"{R:<6} | {mean:>10.4f} | {std:>9.4f} | {ratio:>12.2f}x")

        rows.append(dict(
            experiment=2, N=FIXED_N, M=M, R=R,
            p_bits=P_BITS, mean_s=mean, std_s=std,
        ))

    return rows


def run_experiment_3() -> List[dict]:
    """Full N × R grid."""
    print("\n" + "=" * 55)
    print(f"Experiment 3: Runtime Grid N×R (M={M})")
    print("=" * 55)
    header = f"{'N\\R':<6} | " + " | ".join(f"R={r:2d} (s)" for r in R_GRID)
    print(header)
    print("-" * len(header))

    rows = []
    grid: dict = {}

    for N in N_GRID:
        ballots = make_ballots(N, M)
        row_strs = []
        for R in R_GRID:
            ballot_periods = [ballots] * R
            mean, std = measure(ballot_periods)
            grid[(N, R)] = mean
            row_strs.append(f"{mean:8.4f}")
            rows.append(dict(
                experiment=3, N=N, M=M, R=R,
                p_bits=P_BITS, mean_s=mean, std_s=std,
            ))
        print(f"N={N:<4} | " + " | ".join(row_strs))

    return rows, grid


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_csv(all_rows: List[dict], path: Path) -> None:
    """Write all benchmark results to a CSV file."""
    fieldnames = ["experiment", "N", "M", "R", "p_bits", "mean_s", "std_s"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nResults saved → {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_vary_n(rows1: List[dict]) -> None:
    """Plot 1: Runtime vs N with O(N) reference line."""
    Ns = [r["N"] for r in rows1]
    means = [r["mean_s"] for r in rows1]
    stds = [r["std_s"] for r in rows1]

    # O(N) reference anchored at the first data point
    ref = [means[0] * (n / Ns[0]) for n in Ns]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(Ns, means, yerr=stds, fmt="o-", color="steelblue",
                linewidth=2, markersize=6, capsize=4, label="PP-Phragmén")
    ax.plot(Ns, ref, "--", color="tomato", linewidth=1.5, label="O(N) reference")

    ax.set_xscale("log")
    ax.set_xlabel("Number of Voters (N)", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title("PP-Phragmén Runtime vs Number of Voters", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xticks(Ns)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    path = OUTPUT_DIR / "plot_vary_N.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Plot saved    → {path}")


def plot_vary_r(rows2: List[dict]) -> None:
    """Plot 2: Runtime vs R with O(R) reference line."""
    Rs = [r["R"] for r in rows2]
    means = [r["mean_s"] for r in rows2]
    stds = [r["std_s"] for r in rows2]

    ref = [means[0] * (r / Rs[0]) for r in Rs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(Rs, means, yerr=stds, fmt="s-", color="steelblue",
                linewidth=2, markersize=6, capsize=4, label="PP-Phragmén")
    ax.plot(Rs, ref, "--", color="tomato", linewidth=1.5, label="O(R) reference")

    ax.set_xlabel("Number of Rounds (R)", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title("PP-Phragmén Runtime vs Number of Rounds", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xticks(Rs)

    path = OUTPUT_DIR / "plot_vary_R.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Plot saved    → {path}")


def plot_heatmap(grid: dict) -> None:
    """Plot 3: Heatmap of runtime over N × R grid."""
    data = np.array(
        [[grid[(N, R)] for R in R_GRID] for N in N_GRID],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, aspect="auto", origin="lower", cmap="YlOrRd")

    ax.set_xticks(range(len(R_GRID)))
    ax.set_xticklabels([str(r) for r in R_GRID], fontsize=11)
    ax.set_yticks(range(len(N_GRID)))
    ax.set_yticklabels([str(n) for n in N_GRID], fontsize=11)

    ax.set_xlabel("Number of Rounds (R)", fontsize=12)
    ax.set_ylabel("Number of Voters (N)", fontsize=12)
    ax.set_title(f"PP-Phragmén Runtime Heatmap (M={M})", fontsize=14)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Runtime (seconds)", fontsize=11)

    # Annotate cells with rounded values
    for i, N in enumerate(N_GRID):
        for j, R in enumerate(R_GRID):
            val = grid[(N, R)]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if val < data.max() * 0.7 else "white")

    path = OUTPUT_DIR / "plot_heatmap_NxR.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Plot saved    → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"PP-Phragmén Runtime Benchmark")
    print(f"  M={M}, p=2^{P_BITS}-1, parties={N_PARTIES}, threshold={THRESHOLD}, reps={REPS}")

    rows1 = run_experiment_1()
    rows2 = run_experiment_2()
    rows3, grid = run_experiment_3()

    all_rows = rows1 + rows2 + rows3
    save_csv(all_rows, OUTPUT_DIR / "results.csv")

    print("\nGenerating plots...")
    plot_vary_n(rows1)
    plot_vary_r(rows2)
    plot_heatmap(grid)

    print("\nDone.")


if __name__ == "__main__":
    main()
