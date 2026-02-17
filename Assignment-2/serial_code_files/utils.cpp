#include <math.h>
#include "utils.h"
#include <cstdlib>
// Problem 01
void matrix_multiplication(double** A, double** B, double** C, int N) {
    int i, j, k;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            C[i][j] = 0.0;

    for (i = 0; i < N; i++) {
        for (k = 0; k < N; k++) {
            double aik = A[i][k];
            for (j = 0; j < N; j++) {
                C[i][j] += aik * B[k][j];
            }
        }
    }
}

// Problem 02
void transpose(double** m, double** mt, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            mt[j][i] = m[i][j];
}

void transposed_matrix_multiplication(double** m1, double** m2, double** result, int N) {
    int i, j, k;

    // allocate transpose manually
    double** m2t = new double*[N];
    for (i = 0; i < N; i++)
        m2t[i] = new double[N];

    transpose(m2, m2t, N);

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            result[i][j] = 0.0;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < N; k++)
                sum += m1[i][k] * m2t[j][k];
            result[i][j] = sum;
        }
    }

    for (i = 0; i < N; i++)
        delete[] m2t[i];
    delete[] m2t;
}

// Problem 03
void block_matrix_multiplication(double** m1, double** m2, double** result, int B, int N) {
    int ii, jj, kk, i, j, k;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            result[i][j] = 0.0;

    for (ii = 0; ii < N; ii += B) {
        for (kk = 0; kk < N; kk += B) {
            for (jj = 0; jj < N; jj += B) {

                for (i = ii; i < ii + B; i++) {
                    for (k = kk; k < kk + B; k++) {
                        double aik = m1[i][k];
                        for (j = jj; j < jj + B; j++) {
                            result[i][j] += aik * m2[k][j];
                        }
                    }
                }

            }
        }
    }
}


