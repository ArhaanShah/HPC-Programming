#ifndef UTILS_H
#define UTILS_H
#include <stdlib.h>
#include <time.h>
#include <stdarg.h>
#include "init.h"
extern int NUM_Threads;
static inline void *hpc_aligned_alloc(size_t alignment, size_t size) {
    (void)alignment;
    return malloc(size);
}
#define aligned_alloc hpc_aligned_alloc
int hpc_printf(const char *fmt, ...);
#define printf hpc_printf
void parallel_interpolation(double* mesh_value,Points* points,LocalMeshPad& pad);
void save_mesh(double *mesh_value);

#endif
