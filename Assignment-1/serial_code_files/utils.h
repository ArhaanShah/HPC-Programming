#ifndef UTILS_H
#define UTILS_H
#include <time.h>

void vector_triad_operation(double *x, double *y, double *v, double *S, int Np);
void copy_operation(double *x, double *y, double *v, double *S, int Np);
void scale_operation(double *x, double *y, double *v, double *S, int Np);
void add_operation(double *x, double *y, double *v, double *S, int Np);
void energy_operation(double *x, double *y, double *v, double *S, int Np);
void dummy(int x);

#endif
