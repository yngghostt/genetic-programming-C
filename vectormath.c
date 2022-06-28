#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double **dynamic_array_alloc(size_t N, size_t M)
{
    double **A = (double **)malloc(N*sizeof(double *));
    int i;
    for(i = 0; i < N; i++) {
        A[i] = (double *)malloc(M*sizeof(double ));
    }
    return A;
}

int **int_dynamic_array_alloc(size_t N, size_t M)
{
    int **A = (int **)malloc(N*sizeof(int *));
    int i;
    for(i = 0; i < N; i++) {
        A[i] = (int *)malloc(M*sizeof(int ));
    }
    return A;
}

double **dynamic_array_alloc_zeros(size_t N, size_t M)
{
    double **A = (double **)malloc(N*sizeof(double *));
    int i, j;
    for(i = 0; i < N; i++) {
        A[i] = (double *)malloc(M*sizeof(double));
        for (j = 0; j < M; j++)
            A[i][j] = 0;
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
        printf("%.0f ", x[i]);
    }
    printf("\n-----\n");
}

void matrix_print( double **x, int rows, int cols )
{
    int i, j;
    for(i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%.15f ", x[i][j]);
        }
        printf("\n");
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

double norm( double *x, int n )
{
    return sqrt(dot(x,x,n));
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

void matrix_plus( double **a, double **b, double **res, int rows, int cols )
{
    int i, j;
    for (i = 0; i < rows; i++)
        for(j = 0; j < cols; j++)
            res[i][j] = a[i][j] + b[i][j];
}

void matrix_minus( double **a, double **b, double **res, int rows, int cols )
{
    int i, j;
    for (i = 0; i < rows; i++)
        for(j = 0; j < cols; j++)
            res[i][j] = a[i][j] - b[i][j];
}

void matrix_multiply( double **m, double a, double **res, int rows, int cols )
{
    int i, j;
    for (i = 0; i < rows; i++)
        for(j = 0; j < cols; j++)
            res[i][j] = a*m[i][j];
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
    double max = DBL_MIN;
    for(i = 0; i < n; i++) {
        if (x[i] > max)
            j = i;
        max = x[i];
    }
    return j;
}
