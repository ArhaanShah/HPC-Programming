from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

LAB_DATA = ROOT / "data_lab" / "times_parallel_labpc.csv"
CLUSTER_DATA = ROOT / "data_cluster" / "times_parallel_cluster.csv"

LAB_EXECUTION_PLOT = RESULTS_DIR / "execution_time_vs_threads_lab_workstation.png"
CLUSTER_EXECUTION_PLOT = RESULTS_DIR / "execution_time_vs_threads_cluster.png"
LAB_SPEEDUP_PLOT = RESULTS_DIR / "speedup_vs_threads_lab_workstation.png"
CLUSTER_SPEEDUP_PLOT = RESULTS_DIR / "speedup_vs_threads_cluster.png"
EFFICIENCY_PLOT = RESULTS_DIR / "parallel_efficiency_vs_threads_500x200_20000000_cluster.png"


def plot_execution_time(data_path: Path, output_path: Path, title: str) -> None:
    df = pd.read_csv(data_path)
    plt.figure(figsize=(9, 6))

    for (nx, ny, num_points), group in df.groupby(["nx", "ny", "num_points"]):
        ordered = group.sort_values("threads")
        label = f"{nx}x{ny}, {num_points} points"
        plt.plot(
            ordered["threads"],
            ordered["total_algorithm_time_seconds"],
            marker="o",
            linewidth=2,
            label=label,
        )

    plt.title(title)
    plt.xlabel("Threads")
    plt.ylabel("Total Algorithm Time (s)")
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
        baseline = ordered.loc[ordered["threads"] == 1, "total_algorithm_time_seconds"].iloc[0]
        ordered["speedup"] = baseline / ordered["total_algorithm_time_seconds"]
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


def plot_efficiency(data_path: Path, output_path: Path) -> None:
    df = pd.read_csv(data_path)
    filtered = df[(df["nx"] == 500) & (df["ny"] == 200) & (df["num_points"] == 20000000)].copy()
    filtered = filtered.sort_values("threads")
    baseline = filtered.loc[filtered["threads"] == 1, "total_algorithm_time_seconds"].iloc[0]
    filtered["efficiency"] = (baseline / filtered["total_algorithm_time_seconds"]) / filtered["threads"]

    plt.figure(figsize=(8, 5))
    plt.plot(filtered["threads"], filtered["efficiency"], marker="o", linewidth=2, color="darkgreen")
    plt.title("Assignment 7: Parallel Efficiency vs Threads (500x200, 20M Points, Cluster)")
    plt.xlabel("Threads")
    plt.ylabel("Parallel Efficiency")
    plt.xticks([1, 2, 4, 8, 16])
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    plot_execution_time(
        LAB_DATA,
        LAB_EXECUTION_PLOT,
        "Assignment 7: Total Algorithm Time vs Threads (Lab Workstation)",
    )
    plot_execution_time(
        CLUSTER_DATA,
        CLUSTER_EXECUTION_PLOT,
        "Assignment 7: Total Algorithm Time vs Threads (Cluster)",
    )
    plot_speedup(
        LAB_DATA,
        LAB_SPEEDUP_PLOT,
        "Assignment 7: Total Algorithm Speedup vs Threads (Lab Workstation)",
    )
    plot_speedup(
        CLUSTER_DATA,
        CLUSTER_SPEEDUP_PLOT,
        "Assignment 7: Total Algorithm Speedup vs Threads (Cluster)",
    )
    plot_efficiency(CLUSTER_DATA, EFFICIENCY_PLOT)


if __name__ == "__main__":
    main()
