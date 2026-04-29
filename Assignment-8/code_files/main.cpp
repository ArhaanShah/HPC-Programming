#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

static void compute_particle_partition(int total_point_count, int process_rank,
                                       int process_count, int *start_index,
                                       int *partition_count) {
    int base_partition_size = total_point_count / process_count;
    int remainder_count = total_point_count % process_count;
    *start_index = process_rank * base_partition_size
                 + (process_rank < remainder_count ? process_rank : remainder_count);
    *partition_count = base_partition_size + (process_rank < remainder_count ? 1 : 0);
}

int main(int argc, char **argv) {
    int thread_support_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_support_level);

    int process_rank, process_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    if (argc != 2) {
        if (process_rank == 0) printf("Usage: %s <input_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int input_header[4];
    Points *global_points = NULL;
    int global_point_count = 0;

    if (process_rank == 0) {
        FILE *input_file = fopen(argv[1], "rb");
        if (!input_file) {
            printf("Error opening input file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fread(&input_header[0], sizeof(int), 1, input_file);
        fread(&input_header[1], sizeof(int), 1, input_file);
        fread(&input_header[2], sizeof(int), 1, input_file);
        fread(&input_header[3], sizeof(int), 1, input_file);
        global_point_count = input_header[2];

        global_points = (Points *) calloc(global_point_count, sizeof(Points));
        for (int point_index = 0; point_index < global_point_count; point_index++) {
            fread(&global_points[point_index].x, sizeof(double), 1, input_file);
            fread(&global_points[point_index].y, sizeof(double), 1, input_file);
            global_points[point_index].is_void = false;
        }
        fclose(input_file);
    }

    MPI_Bcast(input_header, 4, MPI_INT, 0, MPI_COMM_WORLD);
    NX = input_header[0];
    NY = input_header[1];
    global_point_count = input_header[2];
    Maxiter = input_header[3];
    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    int local_start_index, local_point_count;
    compute_particle_partition(global_point_count, process_rank, process_count,
                               &local_start_index, &local_point_count);

    int *send_counts = NULL;
    int *send_displacements = NULL;
    if (process_rank == 0) {
        send_counts = (int *) malloc(process_count * sizeof(int));
        send_displacements = (int *) malloc(process_count * sizeof(int));
        for (int destination_rank = 0; destination_rank < process_count; destination_rank++) {
            int destination_start_index, destination_point_count;
            compute_particle_partition(global_point_count, destination_rank, process_count,
                                       &destination_start_index, &destination_point_count);
            send_counts[destination_rank] = destination_point_count;
            send_displacements[destination_rank] = destination_start_index;
        }
    }

    MPI_Datatype point_datatype;
    MPI_Type_contiguous(sizeof(Points), MPI_BYTE, &point_datatype);
    MPI_Type_commit(&point_datatype);

    Points *local_points = (Points *) calloc(local_point_count > 0 ? local_point_count : 1,
                                             sizeof(Points));

    MPI_Scatterv(global_points, send_counts, send_displacements, point_datatype,
                 local_points, local_point_count, point_datatype,
                 0, MPI_COMM_WORLD);

    if (process_rank == 0) {
        free(global_points);
        free(send_counts);
        free(send_displacements);
    }

    NUM_Points = local_point_count;

    double *mesh_values = (double *) calloc(GRID_X * GRID_Y, sizeof(double));

    double total_interpolation_time = 0.0;
    double total_normalization_time = 0.0;
    double total_mover_time = 0.0;
    double total_denormalization_time = 0.0;

    for (int iteration_index = 0; iteration_index < Maxiter; iteration_index++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double interpolation_start_time = MPI_Wtime();

        interpolation(mesh_values, local_points);

        double interpolation_end_time = MPI_Wtime();

        MPI_Allreduce(MPI_IN_PLACE, mesh_values, GRID_X * GRID_Y,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double local_minimum_value, local_maximum_value;
        mesh_minmax(mesh_values, &local_minimum_value, &local_maximum_value);
        double global_minimum_value, global_maximum_value;
        MPI_Allreduce(&local_minimum_value, &global_minimum_value, 1,
                      MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&local_maximum_value, &global_maximum_value, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        normalize_with_minmax(mesh_values, global_minimum_value, global_maximum_value);

        double normalization_end_time = MPI_Wtime();

        mover(mesh_values, local_points);

        double mover_end_time = MPI_Wtime();

        denormalization(mesh_values);

        double denormalization_end_time = MPI_Wtime();

        total_interpolation_time += (interpolation_end_time - interpolation_start_time);
        total_normalization_time += (normalization_end_time - interpolation_end_time);
        total_mover_time += (mover_end_time - normalization_end_time);
        total_denormalization_time += (denormalization_end_time - mover_end_time);
    }

    double reported_interpolation_time, reported_normalization_time;
    double reported_mover_time, reported_denormalization_time;
    MPI_Reduce(&total_interpolation_time, &reported_interpolation_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_normalization_time, &reported_normalization_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_mover_time, &reported_mover_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_denormalization_time, &reported_denormalization_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    long long int local_void_count = void_count(local_points);
    long long int global_void_count = 0;
    MPI_Reduce(&local_void_count, &global_void_count, 1,
               MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (process_rank == 0) {
        save_mesh(mesh_values);
        printf("Total Interpolation Time = %lf seconds\n", reported_interpolation_time);
        printf("Total Normalization Time = %lf seconds\n", reported_normalization_time);
        printf("Total Mover Time = %lf seconds\n", reported_mover_time);
        printf("Total Denormalization Time = %lf seconds\n", reported_denormalization_time);
        printf("Total Algorithm Time = %lf seconds\n",
               reported_interpolation_time + reported_normalization_time
             + reported_mover_time + reported_denormalization_time);
        printf("Total Number of Voids = %lld\n", global_void_count);
    }

    MPI_Type_free(&point_datatype);
    free(local_points);
    free(mesh_values);
    MPI_Finalize();
    return 0;
}
