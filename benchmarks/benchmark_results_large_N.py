"""PP-Phragmén large-N runtime benchmark: T_share / T_protocol split.

N in [100, 500, 1000, 2000, 5000], M=3, R=1, p=2**31-1 (Mersenne prime).
Fits linear regression on T_protocol and extrapolates to N=100,000 and N=1,000,000.

Reference: "Fairness Without Exposure: Privacy-Preserving Phragmén Voting."
"""

import csv
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

from protocol.run_election import _make_ballot_shares, run_election  # noqa: E402

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
M: int = 3
P: int = 2**31 - 1  # Mersenne prime
N_PARTIES: int = 3
THRESHOLD: int = 2
REPS: int = 3

N_SWEEP: List[int] = [100, 500, 1000, 2000, 5000]
EXTRAPOLATE: List[int] = [100_000, 1_000_000]

OUTPUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ballots(n_voters: int, n_candidates: int) -> List[List[int]]:
    """Return N×M ballot where voter i approves candidate (i mod M)."""
    row_templates = [[1 if c == j else 0 for j in range(n_candidates)]
                     for c in range(n_candidates)]
    return [row_templates[i % n_candidates] for i in range(n_voters)]


def time_split(ballots: List[List[int]], seed: int = 0) -> Tuple[float, float]:
    """Return (T_share, T_protocol) in wall-clock seconds.

    T_share  = time for _make_ballot_shares alone.
    T_total  = time for run_election (which includes its own sharing call).
    T_protocol = T_total - T_share.
    """
    t0 = time.perf_counter()
    _make_ballot_shares(ballots, N_PARTIES, THRESHOLD, P)
    t_share = time.perf_counter() - t0

    t0 = time.perf_counter()
    run_election([ballots], p=P, n_parties=N_PARTIES, threshold=THRESHOLD, seed=seed)
    t_total = time.perf_counter() - t0

    return t_share, max(t_total - t_share, 0.0)


def measure(n_voters: int) -> Tuple[float, float, float]:
    """Return (mean_T_share, mean_T_protocol, mean_T_total) over REPS."""
    ballots = make_ballots(n_voters, M)
    ts_list: List[float] = []
    tp_list: List[float] = []
    for k in range(REPS):
        t_s, t_p = time_split(ballots, seed=k)
        ts_list.append(t_s)
        tp_list.append(t_p)
    mean_s = statistics.mean(ts_list)
    mean_p = statistics.mean(tp_list)
    return mean_s, mean_p, mean_s + mean_p


# ---------------------------------------------------------------------------
# Linear fit
# ---------------------------------------------------------------------------

def fit_linear(xs: List[int], ys: List[float]) -> Tuple[float, float, float]:
    """Fit y = a*x + b; return (a, b, R²)."""
    coeffs = np.polyfit(xs, ys, deg=1)
    a, b = float(coeffs[0]), float(coeffs[1])
    y_arr = np.array(ys)
    y_hat = a * np.array(xs) + b
    ss_res = float(np.sum((y_arr - y_hat) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("PP-Phragmén Large-N Benchmark")
    print(f"  M={M}, p=2^31-1={P}, R=1, parties={N_PARTIES}, "
          f"threshold={THRESHOLD}, reps={REPS}\n")

    Ns: List[int] = []
    t_shares: List[float] = []
    t_protocols: List[float] = []
    t_totals: List[float] = []

    print("Collecting timings...")
    for N in N_SWEEP:
        print(f"  N={N:>5} ...", end=" ", flush=True)
        mean_s, mean_p, mean_t = measure(N)
        Ns.append(N)
        t_shares.append(mean_s)
        t_protocols.append(mean_p)
        t_totals.append(mean_t)
        print(f"T_share={mean_s:.4f}s  T_protocol={mean_p:.4f}s  T_total={mean_t:.4f}s")

    # Linear regression on T_protocol
    a, b, r2 = fit_linear(Ns, t_protocols)
    extrap_vals = {n: a * n + b for n in EXTRAPOLATE}

    # ---------------------------------------------------------------------------
    # Print table
    # ---------------------------------------------------------------------------
    print()
    print(f"{'N':>6} | {'T_share(s)':>12} | {'T_protocol(s)':>14} | {'T_total(s)':>11}")
    print("------+------------+---------------+-----------")
    for N, ts, tp, tt in zip(Ns, t_shares, t_protocols, t_totals):
        print(f"{N:>6} | {ts:>12.4f} | {tp:>14.4f} | {tt:>11.4f}")

    print()
    print(f"Linear fit (T_protocol): t(N) = {a:.6e} * N + {b:.6f}  (R^2={r2:.6f})")
    print()
    for n_ext in EXTRAPOLATE:
        t_ext = extrap_vals[n_ext]
        print(f"  Estimated T_protocol({n_ext:>10,}) = {t_ext:.2f} seconds  ({t_ext / 60:.2f} min)")

    # ---------------------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------------------
    csv_path = OUTPUT_DIR / "results_large_N.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["N", "T_share_s", "T_protocol_s", "T_total_s"])
        for N, ts, tp, tt in zip(Ns, t_shares, t_protocols, t_totals):
            writer.writerow([N, f"{ts:.6f}", f"{tp:.6f}", f"{tt:.6f}"])
    print(f"\nCSV saved → {csv_path}")

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    N_dense = np.logspace(np.log10(min(Ns)), np.log10(max(EXTRAPOLATE) * 1.5), 600)
    fit_line = a * N_dense + b

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(Ns, t_protocols, color="blue", zorder=5, s=70,
               label="T_protocol (measured)")
    ax.scatter(Ns, t_shares, color="red", zorder=5, s=70,
               label="T_share (measured)")
    ax.plot(N_dense, fit_line, "--", color="steelblue", linewidth=2,
            label=f"Linear fit T_protocol: {a:.2e}·N + {b:.3f}  (R²={r2:.4f})")

    for n_ext in EXTRAPOLATE:
        t_ext = extrap_vals[n_ext]
        ax.scatter([n_ext], [t_ext], marker="*", s=220, color="darkorange", zorder=6)
        ax.annotate(f"N={n_ext:,}\n≈{t_ext:.0f} s",
                    xy=(n_ext, t_ext), xytext=(8, 4),
                    textcoords="offset points", fontsize=8, color="darkorange")

    ax.set_xscale("log")
    ax.set_xlabel("Number of Voters (N)", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("PP-Phragmen Runtime vs N (M=3, R=1, p=2^31-1)", fontsize=13)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)

    plot_path = OUTPUT_DIR / "plot_large_N.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved → {plot_path}")


if __name__ == "__main__":
    main()
