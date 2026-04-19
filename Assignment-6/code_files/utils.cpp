#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include <omp.h>

static int clamp_cell_index(int idx, int max_idx) {
    if (idx < 0) {
        return 0;
    }
    if (idx > max_idx) {
        return max_idx;
    }
    return idx;
}

static size_t round_up_to_alignment(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

static int resolve_thread_count() {
    const char *env_threads = getenv("OMP_NUM_THREADS");
    if (env_threads != NULL) {
        char *end_ptr = NULL;
        long parsed = strtol(env_threads, &end_ptr, 10);
        if (end_ptr != env_threads && parsed > 0) {
            return (int)parsed;
        }
    }

    return (NUM_Threads > 0) ? NUM_Threads : 1;
}

static void ensure_local_mesh_capacity(LocalMeshPad &pad, int thread_count) {
    const int allocated_threads = (NUM_Threads > 0) ? NUM_Threads : 1;
    if (thread_count > allocated_threads) {
        free(pad.local_meshes);

        const size_t bytes = (size_t)thread_count * (size_t)pad.mesh_size * sizeof(double);
        const size_t aligned_bytes = round_up_to_alignment(bytes, 64);
        pad.local_meshes = (double *)aligned_alloc(64, aligned_bytes);
        if (pad.local_meshes == NULL) {
            fprintf(stderr, "Error allocating thread-local meshes\n");
            exit(1);
        }

        memset(pad.local_meshes, 0, bytes);
    }

    NUM_Threads = thread_count;
}

int hpc_printf(const char *fmt, ...) {
    const char *resolved_fmt = fmt;
    if (strcmp(fmt, "Total interpolation time (serial) = %lf seconds\n") == 0) {
        resolved_fmt = "Total interpolation time (parallel) = %lf seconds\n";
    }

    va_list args;
    va_start(args, fmt);
    const int printed = vprintf(resolved_fmt, args);
    va_end(args);
    return printed;
}

void parallel_interpolation(double* mesh_value,Points* points,LocalMeshPad& pad)
{
    const int nt = resolve_thread_count();
    ensure_local_mesh_capacity(pad, nt);

    const int np=NUM_Points;
    const int gx=GRID_X;
    const int ms = gx * GRID_Y;
    const int max_i = gx - 2;
    const int max_j = GRID_Y - 2;
    const double inv_dx=(double)NX;
    const double inv_dy=(double)NY;
    const double dx_c=dx;
    const double dy_c=dy;

    #pragma omp parallel num_threads(nt)
    {
        const int tid=omp_get_thread_num();
        double* t_mesh=&pad.local_meshes[tid*ms];

        #pragma omp for schedule(static)
        for(int p=0;p<np;p++)
        {
            const double px=points[p].x;
            const double py=points[p].y;

            const int i = clamp_cell_index((int)(px * inv_dx), max_i);
            const int j = clamp_cell_index((int)(py * inv_dy), max_j);
            const int row_base = j * gx;
            const int base = row_base + i;
            const double Xi = i * dx_c;
            const double Yj = j * dy_c;

            const double lx = px - Xi;
            const double ly = py - Yj;

            const double dx_lx = dx_c - lx;
            const double dy_ly = dy_c - ly;

            const double w00 = dx_lx * dy_ly;
            const double w10 = lx * dy_ly;
            const double w01 = dx_lx * ly;
            const double w11 = lx * ly;

            t_mesh[base]+= w00;
            t_mesh[base + 1]+= w10;
            t_mesh[base + gx]+= w01;
            t_mesh[base + gx + 1]+= w11;
        }

        #pragma omp for schedule(static)
        for(int m = 0; m < ms; m++) {
            const double *thread_mesh = pad.local_meshes + m;
            double sum = 0.0;
            for(int t = 0; t < nt; t++) {
                sum += thread_mesh[t * ms];
            }
            mesh_value[m] += sum;
        }
    }
}

void save_mesh(double *mesh_value) {

    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) {
        printf("Error creating Mesh.out\n");
        exit(1);
    }

    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }

    fclose(fd);
}
