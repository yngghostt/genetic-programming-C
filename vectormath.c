#include <limits.h>
#include <stdlib.h>

double **dynamic_array_alloc(size_t N, size_t M)
{
    double **A = (double **)malloc(N*sizeof(double *));
    int i;
    for(i = 0; i < N; i++) {
        A[i] = (double *)malloc(M*sizeof(double ));
    }
    return A;
}

void dynamic_array_free(double **A, size_t N)
{
    int i;
    for(i = 0; i < N; i++) {
        free(A[i]);
    }
    free(A);
}

double dot( double *x, double *y, int n)
{
    double res = 0;
    int i;

    for (i = 0; i < n; i++) {
        res += x[i] * y[i];
    }
    return res;

}

void mxv( double **m,  double *v, double *res, int rows, int cols )
{
    int i;
    for (i = 0; i < rows; i++)
    {
        res[i] = dot(m[i], v, cols);
    }
}

void plus( double *x, double *y, double *res, int n )
{
    int i;
    for (i = 0; i < n; i++)
    {
        res[i] = x[i]+y[i];
    }
}

void minus( double *x, double *y, double *res, int n )
{
    int i;
    for (i = 0; i < n; i++)
    {
        res[i] = x[i]-y[i];
    }
}

void multiply( double *x, double a, double *res, int n )
{
    int i;
    for (i = 0; i < n; i++)
    {
        res[i] = x[i]*a;
    }
}

int max_ind( double *x, int n )
{
    int i, j = 0;
    double min = LONG_MAX;
    for(i = 0; i < n; i++) {
        if (x[i] < min)
            j = i;
    }
    return j;
}