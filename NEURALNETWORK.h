#ifndef NNET_TEST_NEURALNETWORK_H
#define NNET_TEST_NEURALNETWORK_H

#define LAYER1 784
#define LAYER2 100
#define LAYER3 10


typedef struct {
    double *h1_biases;
    double *h2_biases;
    double *o_biases;

    double **h1_weights;
    double **h2_weights;
    double **o_weights;
} NEURALNETWORK;

NEURALNETWORK init( void );
int forward_propagation( NEURALNETWORK *network, double *x );
void update_batch( NEURALNETWORK *network, double *batch, int l_rate );
double backprop( NEURALNETWORK *network, double *x, double *y );
void fit( NEURALNETWORK *network, double data, int epochs, int mini_batch_size, int l_rate );

#endif
