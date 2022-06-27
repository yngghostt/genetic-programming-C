#ifndef NNET_VECTORMATH_H
#define NNET_VECTORMATH_H

double **dynamic_array_alloc( size_t N, size_t M );
void dynamic_array_free( double **A, size_t N );
extern double dot( double *v, double *y, int n );
extern void mxv(  double **m,  double *v,double *res, int rows, int cols );
void plus( double *x, double *y, double *res, int n );
void minus( double *x, double *y, double *res, int n );
void multiply( double *x, double a, double *res, int n );
int max_ind(double *x, int n );

#endif
