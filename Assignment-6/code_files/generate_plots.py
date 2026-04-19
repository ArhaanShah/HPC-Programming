from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

LAB_DATA = ROOT / "data_lab" / "times_parallel.csv"
CLUSTER_DATA = ROOT / "data_cluster" / "times_parallel.csv"

LAB_EXECUTION_PLOT = RESULTS_DIR / "execution_time_vs_threads_lab_workstation.png"
CLUSTER_EXECUTION_PLOT = RESULTS_DIR / "execution_time_vs_threads_cluster.png"
LAB_SPEEDUP_PLOT = RESULTS_DIR / "speedup_vs_threads_lab_workstation.png"
CLUSTER_SPEEDUP_PLOT = RESULTS_DIR / "speedup_vs_threads_cluster.png"


def plot_execution_time(data_path: Path, output_path: Path, title: str) -> None:
    df = pd.read_csv(data_path)
    plt.figure(figsize=(9, 6))

    for (nx, ny, num_points), group in df.groupby(["nx", "ny", "num_points"]):
        ordered = group.sort_values("threads")
        label = f"{nx}x{ny}, {num_points} points"
        plt.plot(
            ordered["threads"],
            ordered["total_interpolation_time_seconds"],
            marker="o",
            linewidth=2,
            label=label,
        )

    plt.title(title)
    plt.xlabel("Threads")
    plt.ylabel("Interpolation Time (s)")
    plt.xticks([1, 2, 4, 8, 16])
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_speedup(data_path: Path, output_path: Path, title: str) -> None:
    df = pd.read_csv(data_path)
    plt.figure(figsize=(9, 6))

    for (nx, ny, num_points), group in df.groupby(["nx", "ny", "num_points"]):
        ordered = group.sort_values("threads").copy()
        baseline = ordered.loc[ordered["threads"] == 1, "total_interpolation_time_seconds"].iloc[0]
        ordered["speedup"] = baseline / ordered["total_interpolation_time_seconds"]
        label = f"{nx}x{ny}, {num_points} points"
        plt.plot(ordered["threads"], ordered["speedup"], marker="o", linewidth=2, label=label)

    plt.title(title)
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.xticks([1, 2, 4, 8, 16])
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    plot_execution_time(
        LAB_DATA,
        LAB_EXECUTION_PLOT,
        "Assignment 6: Interpolation Execution Time vs Threads (Lab Workstation)",
    )
    plot_execution_time(
        CLUSTER_DATA,
        CLUSTER_EXECUTION_PLOT,
        "Assignment 6: Interpolation Execution Time vs Threads (Cluster)",
    )
    plot_speedup(
        LAB_DATA,
        LAB_SPEEDUP_PLOT,
        "Assignment 6: Interpolation Speedup vs Threads (Lab Workstation)",
    )
    plot_speedup(
        CLUSTER_DATA,
        CLUSTER_SPEEDUP_PLOT,
        "Assignment 6: Interpolation Speedup vs Threads (Cluster)",
    )


if __name__ == "__main__":
    main()
