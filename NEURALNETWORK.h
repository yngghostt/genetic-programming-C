#ifndef NNET_NEURALNETWORK_H
#define NNET_NEURALNETWORK_H

#define LAYER1 784
#define LAYER2 50
#define LAYER3 10


typedef struct {
    double *h1_biases;
    double *h2_biases;
    double *o_biases;

    double **h1_weights;
    double **h2_weights;
    double **o_weights;
} NEURALNETWORK;

typedef struct {
    double *h1_del_b;
    double *h2_del_b;
    double *o_del_b;

    double **h1_del_w;
    double **h2_del_w;
    double **o_del_w;

} DELTA;

NEURALNETWORK init( void );
double sigmoid( double x );
int forward_propagation( NEURALNETWORK *network, double *x );
void back_propagation( NEURALNETWORK *network, DELTA *delta, double *x, double *y );
void update_batch( NEURALNETWORK *network, double **batch_x, double **batch_y, int start, int size, double l_rate );
void fit( NEURALNETWORK *network, double **data_x, double **data_y, long int data_size, int epochs, int mini_batch_size, double l_rate );
double accuracy(NEURALNETWORK *network, double **test_x, double **test_y, int start, int test_size );
#endif
