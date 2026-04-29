from __future__ import print_function
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time

LAB_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIRECTORY = os.path.join(LAB_DIRECTORY, "outputs")
CLUSTER_DATA_DIRECTORY = os.path.join(os.path.dirname(LAB_DIRECTORY), "data_cluster")
SUMMARY_CSV_PATH = os.path.join(CLUSTER_DATA_DIRECTORY, "times_parallel_cluster.csv")
HOSTFILE_PATH = os.path.join(LAB_DIRECTORY, "hostfile")
CLUSTER_HOSTS = ["gics1", "gics2", "gics3", "gics4"]

MPI_INSTALLATION_PREFIX = "/usr/mpi/gcc/openmpi-1.8.8"
EXECUTION_ENVIRONMENT = os.environ.copy()
EXECUTION_ENVIRONMENT["PATH"] = (
    MPI_INSTALLATION_PREFIX + "/bin:" + EXECUTION_ENVIRONMENT.get("PATH", "")
)
EXECUTION_ENVIRONMENT["LD_LIBRARY_PATH"] = (
    MPI_INSTALLATION_PREFIX + "/lib64:"
    + EXECUTION_ENVIRONMENT.get("LD_LIBRARY_PATH", "")
)

CONFIGURATIONS = {
    "a": dict(grid_x_cells=250, grid_y_cells=100, point_count=900000, iteration_count=10),
    "b": dict(grid_x_cells=250, grid_y_cells=100, point_count=5000000, iteration_count=10),
    "c": dict(grid_x_cells=500, grid_y_cells=200, point_count=3600000, iteration_count=10),
    "d": dict(grid_x_cells=500, grid_y_cells=200, point_count=20000000, iteration_count=10),
    "e": dict(grid_x_cells=1000, grid_y_cells=400, point_count=14000000, iteration_count=10),
}

CORE_LAYOUT_BY_TOTAL_CORES = {
    2: (1, 2, 1),
    4: (1, 4, 1),
    8: (1, 8, 1),
    16: (2, 8, 1),
    32: (4, 8, 2),
    64: (8, 8, 4),
}

ABSOLUTE_TOLERANCE = 5e-6
RELATIVE_TOLERANCE = 1e-6

CSV_FIELD_NAMES = [
    "config",
    "total_cores",
    "ranks",
    "threads",
    "interpolation_time_seconds",
    "normalization_time_seconds",
    "mover_time_seconds",
    "denormalization_time_seconds",
    "total_algorithm_time_seconds",
    "void_count",
    "wall_time_seconds",
    "max_absolute_difference",
    "max_relative_difference",
    "speedup",
    "efficiency_percent",
]

TIME_PATTERNS = {
    "interpolation_time_seconds": re.compile(r"Total Interpolation Time = ([\d.]+)"),
    "normalization_time_seconds": re.compile(r"Total Normalization Time = ([\d.]+)"),
    "mover_time_seconds": re.compile(r"Total Mover Time = ([\d.]+)"),
    "denormalization_time_seconds": re.compile(r"Total Denormalization Time = ([\d.]+)"),
    "total_algorithm_time_seconds": re.compile(r"Total Algorithm Time = ([\d.]+)"),
    "void_count": re.compile(r"Total Number of Voids = (-?\d+)"),
}


def run_shell_command(command, working_directory=None, environment=None, capture_output=True):
    if environment is None:
        environment = EXECUTION_ENVIRONMENT
    if isinstance(command, list):
        command = " ".join(command)
    process = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=working_directory or LAB_DIRECTORY,
        env=environment,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
    )
    output_text, _ = process.communicate()
    if output_text is None:
        output_text = ""
    if not isinstance(output_text, str):
        output_text = output_text.decode("utf-8", "replace")
    return process.returncode, output_text


def write_hostfile(host_count):
    with open(HOSTFILE_PATH, "w") as hostfile:
        for host_name in CLUSTER_HOSTS[:host_count]:
            hostfile.write("%s slots=16\n" % host_name)


def compile_binaries():
    print("[compile] building binaries...")
    return_code, output_text = run_shell_command(
        "g++ -O3 -std=c++11 input_file_maker.cpp -o input_maker.out"
    )
    if return_code != 0:
        print(output_text)
        sys.exit("input_file_maker compile failed")

    return_code, output_text = run_shell_command(
        "mpic++ -O3 -std=c++11 -fopenmp -mavx2 -DNDEBUG "
        "main.cpp utils.cpp init.cpp -lm -o main_parallel.out"
    )
    if return_code != 0:
        print(output_text)
        sys.exit("main_parallel compile failed")
    print("[compile] OK")


def generate_input_file(configuration):
    command = "./input_maker.out %d %d %d %d" % (
        configuration["grid_x_cells"],
        configuration["grid_y_cells"],
        configuration["point_count"],
        configuration["iteration_count"],
    )
    return_code, output_text = run_shell_command(command)
    if return_code != 0:
        print(output_text)
        sys.exit("input generation failed")


def build_parallel_command(total_core_count):
    rank_count, thread_count, host_count = CORE_LAYOUT_BY_TOTAL_CORES[total_core_count]
    write_hostfile(host_count)

    if rank_count == 1:
        mapping_options = "-bind-to socket"
    elif rank_count == 2:
        mapping_options = "-map-by ppr:1:socket -bind-to socket"
    else:
        mapping_options = "-map-by ppr:2:node -bind-to socket"

    return (
        "OMP_NUM_THREADS=%d OMP_PROC_BIND=close OMP_PLACES=cores "
        "mpirun --hostfile %s -np %d %s "
        "-x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x LD_LIBRARY_PATH "
        "./main_parallel.out input.bin 2>&1"
    ) % (thread_count, HOSTFILE_PATH, rank_count, mapping_options)


def run_parallel_benchmark(total_core_count, log_path):
    command = build_parallel_command(total_core_count)
    best_return_code = -1
    best_output_text = ""
    best_wall_time = 0.0
    best_algorithm_time = float("inf")

    for repetition_index in range(3):
        start_time = time.time()
        return_code, output_text = run_shell_command(command)
        wall_time = time.time() - start_time
        if return_code != 0:
            best_return_code = return_code
            best_output_text = output_text
            best_wall_time = wall_time
            continue

        algorithm_match = TIME_PATTERNS["total_algorithm_time_seconds"].search(output_text)
        algorithm_time = float(algorithm_match.group(1)) if algorithm_match else float("inf")
        if algorithm_time < best_algorithm_time:
            best_algorithm_time = algorithm_time
            best_return_code = return_code
            best_output_text = output_text
            best_wall_time = wall_time

    with open(log_path, "w") as log_file:
        log_file.write(
            "# command: %s (best of 3)\n# wall=%.4fs return_code=%d\n%s"
            % (command, best_wall_time, best_return_code, best_output_text)
        )
    return best_return_code, best_output_text, best_wall_time


def parse_timing_output(output_text):
    timing_values = {}
    for field_name, pattern in TIME_PATTERNS.items():
        match = pattern.search(output_text)
        if match:
            if field_name == "void_count":
                timing_values[field_name] = int(match.group(1))
            else:
                timing_values[field_name] = float(match.group(1))
    return timing_values


def compare_mesh_outputs(reference_path, current_path):
    with open(reference_path) as reference_file:
        reference_values = reference_file.read().split()
    with open(current_path) as current_file:
        current_values = current_file.read().split()

    if len(reference_values) != len(current_values):
        return "length_mismatch", len(reference_values), len(current_values)

    maximum_absolute_difference = 0.0
    maximum_relative_difference = 0.0
    for reference_value, current_value in zip(reference_values, current_values):
        reference_number = float(reference_value)
        current_number = float(current_value)
        absolute_difference = abs(reference_number - current_number)
        if absolute_difference > maximum_absolute_difference:
            maximum_absolute_difference = absolute_difference
        denominator = max(abs(reference_number), abs(current_number), 1e-12)
        relative_difference = absolute_difference / denominator
        if relative_difference > maximum_relative_difference:
            maximum_relative_difference = relative_difference
    return maximum_absolute_difference, maximum_relative_difference, len(reference_values)


def run_serial_baseline(configuration_id, output_directory):
    write_hostfile(1)
    command = (
        "OMP_NUM_THREADS=1 mpirun --hostfile %s -np 1 "
        "-x OMP_NUM_THREADS -x LD_LIBRARY_PATH "
        "./main_parallel.out input.bin 2>&1"
    ) % HOSTFILE_PATH

    start_time = time.time()
    return_code, output_text = run_shell_command(command)
    wall_time = time.time() - start_time

    with open(os.path.join(output_directory, "serial.log"), "w") as log_file:
        log_file.write(
            "# command: %s\n# wall=%.4fs return_code=%d\n%s"
            % (command, wall_time, return_code, output_text)
        )

    if return_code != 0:
        print(output_text)
        sys.exit("serial run failed for %s" % configuration_id)

    timing_values = parse_timing_output(output_text)
    print(
        "[serial] total=%.4fs interpolation=%.4fs mover=%.4fs voids=%d wall=%.2fs"
        % (
            timing_values["total_algorithm_time_seconds"],
            timing_values["interpolation_time_seconds"],
            timing_values["mover_time_seconds"],
            timing_values["void_count"],
            wall_time,
        )
    )
    return timing_values, wall_time


def run_configuration(configuration_id, total_core_counts):
    configuration = CONFIGURATIONS[configuration_id]
    print("\n" + "=" * 72)
    print(
        "Configuration %s: NX=%d NY=%d points=%d maxiter=%d"
        % (
            configuration_id,
            configuration["grid_x_cells"],
            configuration["grid_y_cells"],
            configuration["point_count"],
            configuration["iteration_count"],
        )
    )
    print("=" * 72)

    configuration_output_directory = os.path.join(
        OUTPUT_DIRECTORY, "config_" + configuration_id
    )
    if not os.path.isdir(configuration_output_directory):
        os.makedirs(configuration_output_directory)

    print("[input] generating input.bin (%d points)..." % configuration["point_count"])
    generate_input_file(configuration)
    input_size_bytes = os.path.getsize(os.path.join(LAB_DIRECTORY, "input.bin"))
    print("[input] input.bin = %.1f MB" % (input_size_bytes / 1024.0 / 1024.0))

    print("[run] serial baseline (np=1, OMP=1)...")
    serial_timing_values, serial_wall_time = run_serial_baseline(
        configuration_id, configuration_output_directory
    )

    reference_mesh_path = os.path.join(configuration_output_directory, "Mesh_serial.out")
    shutil.copyfile(os.path.join(LAB_DIRECTORY, "Mesh.out"), reference_mesh_path)

    configuration_rows = []
    serial_row = dict(config=configuration_id, total_cores=1, ranks=1, threads=1)
    serial_row.update(serial_timing_values)
    serial_row["wall_time_seconds"] = serial_wall_time
    serial_row["max_absolute_difference"] = 0.0
    serial_row["max_relative_difference"] = 0.0
    serial_row["speedup"] = 1.0
    serial_row["efficiency_percent"] = 100.0
    configuration_rows.append(serial_row)

    for total_core_count in total_core_counts:
        rank_count, thread_count, host_count = CORE_LAYOUT_BY_TOTAL_CORES[total_core_count]
        log_path = os.path.join(
            configuration_output_directory, "p%d_t%d.log" % (rank_count, thread_count)
        )
        print(
            "[run] %d cores (np=%d threads/rank=%d hosts=%d)..."
            % (total_core_count, rank_count, thread_count, host_count)
        )

        return_code, output_text, wall_time = run_parallel_benchmark(
            total_core_count, log_path
        )
        if return_code != 0:
            print("[fail] cores=%d:" % total_core_count)
            print(output_text)
            continue

        timing_values = parse_timing_output(output_text)
        current_mesh_path = os.path.join(LAB_DIRECTORY, "Mesh.out")
        comparison_result = compare_mesh_outputs(reference_mesh_path, current_mesh_path)
        if comparison_result[0] == "length_mismatch":
            print(
                "    Length mismatch: reference=%d current=%d"
                % (comparison_result[1], comparison_result[2])
            )
            maximum_absolute_difference = float("nan")
            maximum_relative_difference = float("nan")
            validation_label = "MISMATCH"
        else:
            maximum_absolute_difference, maximum_relative_difference, _ = comparison_result
            is_valid = (
                maximum_absolute_difference <= ABSOLUTE_TOLERANCE
                and maximum_relative_difference <= RELATIVE_TOLERANCE
            )
            validation_label = "OK" if is_valid else "MISMATCH"

        speedup = (
            serial_timing_values["total_algorithm_time_seconds"]
            / timing_values["total_algorithm_time_seconds"]
            if timing_values.get("total_algorithm_time_seconds", 0) > 0
            else 0.0
        )
        efficiency_percent = speedup / total_core_count * 100.0

        print(
            "    total=%.4fs speedup=%.2fx efficiency=%.1f%% "
            "interpolation=%.4fs mover=%.4fs difference(abs/rel)=%.2e/%.2e %s"
            % (
                timing_values.get("total_algorithm_time_seconds", 0.0),
                speedup,
                efficiency_percent,
                timing_values.get("interpolation_time_seconds", 0.0),
                timing_values.get("mover_time_seconds", 0.0),
                maximum_absolute_difference,
                maximum_relative_difference,
                validation_label,
            )
        )

        row = dict(
            config=configuration_id,
            total_cores=total_core_count,
            ranks=rank_count,
            threads=thread_count,
            wall_time_seconds=wall_time,
            max_absolute_difference=maximum_absolute_difference,
            max_relative_difference=maximum_relative_difference,
            speedup=speedup,
            efficiency_percent=efficiency_percent,
        )
        row.update(timing_values)
        configuration_rows.append(row)

    try:
        os.remove(os.path.join(LAB_DIRECTORY, "input.bin"))
    except OSError:
        pass

    return configuration_rows


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="a,b,c,d,e", help="comma list")
    parser.add_argument("--cores", default="2,4,8,16,32,64", help="comma list")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip recompilation (assume binaries built)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="reduced sweep: configs=a,c cores=2,8,32",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if arguments.quick:
        arguments.configs = "a,c"
        arguments.cores = "2,8,32"

    configuration_ids = [
        config_id.strip() for config_id in arguments.configs.split(",") if config_id.strip()
    ]
    total_core_counts = [
        int(core_count.strip()) for core_count in arguments.cores.split(",") if core_count.strip()
    ]

    for total_core_count in total_core_counts:
        if total_core_count not in CORE_LAYOUT_BY_TOTAL_CORES:
            sys.exit(
                "unknown core count %d (valid: %s)"
                % (total_core_count, sorted(CORE_LAYOUT_BY_TOTAL_CORES))
            )

    if not os.path.isdir(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    if not os.path.isdir(CLUSTER_DATA_DIRECTORY):
        os.makedirs(CLUSTER_DATA_DIRECTORY)

    if not arguments.no_compile:
        compile_binaries()

    all_rows = []
    for configuration_id in configuration_ids:
        if configuration_id not in CONFIGURATIONS:
            print("Skipping unknown configuration %s" % configuration_id)
            continue
        all_rows.extend(run_configuration(configuration_id, total_core_counts))

    with open(SUMMARY_CSV_PATH, "w") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=CSV_FIELD_NAMES)
        writer.writeheader()
        for row in all_rows:
            for field_name in CSV_FIELD_NAMES:
                row.setdefault(field_name, "")
            writer.writerow(row)

    print("\n[summary] %s (%d rows)" % (SUMMARY_CSV_PATH, len(all_rows)))
    print("\n[done] outputs/ and data_cluster/ ready.")


if __name__ == "__main__":
    main()
