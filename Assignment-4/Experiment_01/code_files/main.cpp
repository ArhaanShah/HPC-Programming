#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(void) {
    omp_set_num_threads(4);

    int grid_nx[3] = {250, 500, 1000};
    int grid_ny[3] = {100, 200,  400};

    long particle_counts[5] = {100L, 10000L, 1000000L, 100000000L, 1000000000L};

    Maxiter = 10;

    FILE *fp = fopen("scaling_results.csv", "w");
    if (!fp) { printf("Can't open file\n"); return 1; }
    fprintf(fp, "Grid_NX,Grid_NY,Particles,Total_Interp_s\n");

    printf("EXP 1 SCALING\n");
    printf("Grid    Particles Total\n");

    for (int g = 0; g < 3; g++) {
        NX = grid_nx[g]; NY = grid_ny[g];
        GRID_X = NX + 1; GRID_Y = NY + 1;
        dx = 1.0 / NX; dy = 1.0 / NY;

        double *mesh_value = (double *)calloc((size_t)GRID_X * GRID_Y, sizeof(double));
        if (!mesh_value) { printf("Cannot allocate mesh\n"); continue; }

        for (int c = 0; c < 5; c++) {
            NUM_Points = (int)particle_counts[c];

            Points *points = (Points *)calloc(particle_counts[c], sizeof(Points));
            if (!points) {
                printf("  Skip N=%ld\n", particle_counts[c]);
                fprintf(fp, "%d,%d,%ld,ALLOC_FAIL\n", NX, NY, particle_counts[c]);
                continue;
            }

            initializepoints(points);
            memset(mesh_value, 0, (size_t)GRID_X * GRID_Y * sizeof(double));

            double total_interp = 0.0;
            for (int iter = 0; iter < Maxiter; iter++) {
                clock_t t0 = clock();
                interpolation(mesh_value, points);
                clock_t t1 = clock();
                total_interp += (double)(t1 - t0) / CLOCKS_PER_SEC;
            }

            const char *gname = (NX==250 ? "250x100" : NX==500 ? "500x200" : "1000x400");
            printf("%s %ld %.6f\n", gname, particle_counts[c], total_interp);
            fprintf(fp, "%d,%d,%ld,%.6f\n", NX, NY, particle_counts[c], total_interp);

            free(points);
        }
        free(mesh_value);
    }

    fclose(fp);
    printf("Done. See scaling_results.csv\n");
    return 0;
}
