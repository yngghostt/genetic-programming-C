#ifndef NNET_VECTORMATH_H
#define NNET_VECTORMATH_H

extern double **dynamic_array_alloc( size_t N, size_t M );
extern int **int_dynamic_array_alloc(size_t N, size_t M);
extern double **dynamic_array_alloc_zeros(size_t N, size_t M);
extern void dynamic_array_free( double **A, size_t N );
extern void print( double *x, int n );
void matrix_print( double **x, int rows, int cols );
extern double dot( double *v, double *y, int n );
extern double norm( double *x, int n );
extern void mxv(  double **m,  double *v,double *res, int rows, int cols );
extern void vxv( double *x, double *y, double **res, int n, int m );
extern void plus( double *x, double *y, double *res, int n );
extern void minus( double *x, double *y, double *res, int n );
extern void multiply( double *x, double a, double *res, int n );
extern void matrix_plus( double **a, double **b, double **res, int rows, int cols );
extern void matrix_minus( double **a, double **b, double **res, int rows, int cols );
extern void matrix_multiply( double **m, double a, double **res, int rows, int cols );
extern double **transpose( double **m, int rows, int columns );
extern int max_ind( double *x, int n );


#endif
