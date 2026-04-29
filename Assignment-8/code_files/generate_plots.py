#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Assignment 8 plots from the consolidated timing CSV.

Usage:
  python generate_plots.py
  python generate_plots.py --input ../data_cluster/times_parallel_cluster.csv --out-dir ../results
"""

from __future__ import print_function
import argparse
import csv
import os
import sys


LAB_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_PATH = os.path.join(os.path.dirname(LAB_DIRECTORY), "data_cluster",
                                  "times_parallel_cluster.csv")
DEFAULT_OUTPUT_DIRECTORY = os.path.join(os.path.dirname(LAB_DIRECTORY), "results")
PLOT_CORE_COUNTS = [2, 4, 8, 16, 32, 64]
SEGMENT_CORE_COUNTS = [1, 2, 4, 8, 16, 32, 64]
CONFIGURATION_ORDER = ["a", "b", "c", "d", "e"]
SEGMENT_FIELDS = [
    ("interpolation_time_seconds", "interp", "Interpolation"),
    ("normalization_time_seconds", "norm", "Normalization"),
    ("mover_time_seconds", "mover", "Mover"),
    ("denormalization_time_seconds", "denorm", "Denormalization"),
]


def _float_or_none(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except ValueError:
        return None


def _first_value(row, names):
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return None


def _set_log2_xaxis(plt):
    try:
        plt.xscale("log", base=2)
    except TypeError:
        plt.xscale("log", basex=2)
    plt.xticks(PLOT_CORE_COUNTS, [str(core_count) for core_count in PLOT_CORE_COUNTS])


def load_rows(path):
    if not os.path.exists(path):
        sys.exit("input CSV not found: %s" % path)

    rows_by_configuration = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            configuration_id = row.get("config", "").strip()
            if not configuration_id:
                continue
            rows_by_configuration.setdefault(configuration_id, []).append(row)
    return rows_by_configuration


def normalize_rows(rows_by_configuration):
    normalized = {}
    for configuration_id, rows in rows_by_configuration.items():
        serial_total = None
        parsed = []
        for row in rows:
            try:
                cores = int(row.get("total_cores", ""))
            except ValueError:
                continue
            total = _float_or_none(_first_value(
                row, ["total_algorithm_time_seconds", "total"]
            ))
            speedup = _float_or_none(row.get("speedup"))
            if cores == 1:
                serial_total = total
            parsed.append(dict(cores=cores, total=total, speedup=speedup))

        out = []
        for row in parsed:
            if row["cores"] not in PLOT_CORE_COUNTS or row["total"] is None:
                continue
            speedup = row["speedup"]
            if speedup is None and serial_total and row["total"] > 0.0:
                speedup = serial_total / row["total"]
            if speedup is None:
                continue
            out.append(dict(cores=row["cores"], total=row["total"],
                            speedup=speedup))
        normalized[configuration_id] = sorted(out, key=lambda r: r["cores"])
    return normalized


def normalize_segment_rows(rows_by_configuration):
    normalized = {}
    for configuration_id, rows in rows_by_configuration.items():
        parsed = []
        for row in rows:
            try:
                cores = int(row.get("total_cores", ""))
            except ValueError:
                continue
            if cores not in SEGMENT_CORE_COUNTS:
                continue

            segment_values = {}
            all_segments_present = True
            for formal_name, legacy_name, label in SEGMENT_FIELDS:
                value = _float_or_none(_first_value(row, [formal_name, legacy_name]))
                if value is None:
                    all_segments_present = False
                    break
                segment_values[label] = value

            if all_segments_present:
                segment_values["cores"] = cores
                parsed.append(segment_values)

        normalized[configuration_id] = sorted(parsed, key=lambda r: r["cores"])
    return normalized


def plot_segment_graphs(plt, segment_rows_by_configuration, output_directory):
    colors = ["#4477aa", "#228833", "#cc6677", "#aa3377"]
    segment_paths = []

    for configuration_id in CONFIGURATION_ORDER:
        rows = segment_rows_by_configuration.get(configuration_id, [])
        if not rows:
            continue

        plt.figure(figsize=(8.5, 5.5))
        core_counts = [row["cores"] for row in rows]

        for color_index, (_, _, label) in enumerate(SEGMENT_FIELDS):
            values = [row[label] for row in rows]
            plt.plot(core_counts, values, marker="o", linewidth=2,
                     color=colors[color_index], label=label)

        try:
            plt.xscale("log", base=2)
        except TypeError:
            plt.xscale("log", basex=2)
        plt.yscale("log")
        plt.xticks(core_counts, [str(core_count) for core_count in core_counts])
        plt.xlabel("Total cores")
        plt.ylabel("Segment time (s)")
        plt.title("Config %s Segment Times" % configuration_id)
        plt.grid(True, which="both", linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()

        segment_path = os.path.join(
            output_directory, "segment_times_config_%s.png" % configuration_id
        )
        plt.savefig(segment_path, dpi=140)
        plt.close()
        segment_paths.append(segment_path)

    return segment_paths


def make_plots(rows_by_configuration, segment_rows_by_configuration, output_directory):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        sys.exit("matplotlib is required to generate plots: %s" % exc)

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    plt.figure(figsize=(8, 5.5))
    for configuration_id in CONFIGURATION_ORDER:
        rows = rows_by_configuration.get(configuration_id, [])
        if not rows:
            continue
        plt.plot([r["cores"] for r in rows],
                 [r["speedup"] for r in rows],
                 marker="o", linewidth=2, label="Config %s" % configuration_id)
    plt.plot(PLOT_CORE_COUNTS, PLOT_CORE_COUNTS, "k--", alpha=0.45, label="Ideal")
    _set_log2_xaxis(plt)
    plt.xlabel("Total cores")
    plt.ylabel("Speedup")
    plt.title("Speedup vs Cores")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    speedup_path = os.path.join(output_directory, "speedup_vs_cores.png")
    plt.savefig(speedup_path, dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5.5))
    for configuration_id in CONFIGURATION_ORDER:
        rows = rows_by_configuration.get(configuration_id, [])
        if not rows:
            continue
        plt.plot([r["cores"] for r in rows],
                 [r["total"] for r in rows],
                 marker="o", linewidth=2, label="Config %s" % configuration_id)
    _set_log2_xaxis(plt)
    plt.yscale("log")
    plt.xlabel("Total cores")
    plt.ylabel("Execution time (s)")
    plt.title("Execution Time vs Cores")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    time_path = os.path.join(output_directory, "execution_time_vs_cores.png")
    plt.savefig(time_path, dpi=140)
    plt.close()

    segment_paths = plot_segment_graphs(
        plt, segment_rows_by_configuration, output_directory
    )

    return speedup_path, time_path, segment_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH,
                        help="timing CSV path")
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIRECTORY,
                        help="directory for generated PNG files")
    args = parser.parse_args()

    raw_rows = load_rows(args.input)
    rows = normalize_rows(raw_rows)
    segment_rows = normalize_segment_rows(raw_rows)
    speedup_path, time_path, segment_paths = make_plots(rows, segment_rows, args.out_dir)
    print("[plot] wrote %s" % speedup_path)
    print("[plot] wrote %s" % time_path)
    for segment_path in segment_paths:
        print("[plot] wrote %s" % segment_path)


if __name__ == "__main__":
    main()
