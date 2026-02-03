#include <math.h>
#include "utils.h"

void vector_triad_operation(double *x, double *y, double *v, double *S, int Np) {

    for (int p = 0; p < Np; p++) {
        S[p] = x[p] + v[p] * y[p];

        // Prevent compiler from optimizing away the loop
        if (((double)p) == 333.333)
            dummy(p);

    }
}

void copy_operation(double *x, double *y, double *v, double *S, int Np) {

    for (int p = 0; p < Np; p++) {
        x[p] = y[p];

        if (((double)p) == 333.333)
            dummy(p);

    }
}

void scale_operation(double *x, double *y, double *v, double *S, int Np) {

    for (int p = 0; p < Np; p++) {
        x[p] = v[p] * y[p];

        // Prevent compiler from optimizing away the loop
        if (((double)p) == 333.333)
            dummy(p);

    }
}

void add_operation(double *x, double *y, double *v, double *S, int Np) {

    for (int p = 0; p < Np; p++) {
        S[p] = x[p] + y[p];

        // Prevent compiler from optimizing away the loop
        if (((double)p) == 333.333)
            dummy(p);

    }
}

void energy_operation(double *x, double *y, double *v, double *S, int Np) {

    for (int p = 0; p < Np; p++) {
        S[p] = 0.5*2*(v[p]*v[p]);

        // Prevent compiler from optimizing away the loop
        if (((double)p) == 333.333)
            dummy(p);

    }
}

void dummy(int x) {
    x = 10 * sin(x / 10.0);
}
