#include <limits.h>
#include <stdio.h>
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

void print( double *x, int n )
{
    int i;
    for(i = 0; i < n; i++)
    {
        printf("%.10f ", x[i]);
    }
    printf("\n-----\n");
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

void vxv( double *x, double *y, double **res, int n, int m )
{
    int i, j;
    for (i = 0; i < n; i++)
        for(j = 0; j < m; j++)
            res[i][j] = x[i]*y[j];
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
double **transpose( double **m, int rows, int columns )
{
    double **res = dynamic_array_alloc(columns, rows);
    int i, j;
    for(i = 0; i < rows; i++)
        for(j = 0; j < columns; j++)
            res[j][i] = m[i][j];

    return res;
}

int max_ind( double *x, int n )
{
    int i, j = 0;
    double max = 0;
    for(i = 0; i < n; i++) {
        if (x[i] > max)
            j = i;
        max = x[i];
    }
    return j;
}
