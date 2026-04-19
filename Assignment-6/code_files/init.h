#ifndef INIT_H
#define INIT_H

#include <stdio.h>

// Point structure
typedef struct {
    double x, y;
} Points;

typedef struct{
    int base_idx;
    float w00, w10, w01, w11;
}WorkPoint;

typedef struct{
    int* bin_counts;
    int* bin_offsets;
    int* write_cursors;
    WorkPoint* sorted_work;
    int* local_bin_storage; // For thread-local counting
}ScratchPad;

typedef struct {
    // A single contiguous block: [NUM_Threads * GRID_X * GRID_Y]
    double* local_meshes; 
    int mesh_size; // GRID_X * GRID_Y
} LocalMeshPad;

// Global simulation parameters
extern int GRID_X, GRID_Y, NX, NY;
extern int NUM_Points, Maxiter;
extern double dx, dy;

// Initialization & I/O
void read_points(FILE *file, Points *points);
void initializepoints(Points *points);

#endif