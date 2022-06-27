#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "NEURALNETWORK.h"
#include "vectormath.h"

double sigmoid( double x )
{
    return 1/(1 + exp(-1*x));
}

double dsigmoid( double x )
{
    return sigmoid(x)*(1 - sigmoid(x));
}

double relu( double x )
{
    if (x > 0) {
        return x;
    } else {
        return 0;
    }
}

NEURALNETWORK init( void )
{
    double h1_biases[LAYER2];
    double h2_biases[LAYER2];
    double o_biases[LAYER3];

    double **h1_weights = dynamic_array_alloc(LAYER2, LAYER1);
    double **h2_weights = dynamic_array_alloc(LAYER2, LAYER2);
    double **o_weights = dynamic_array_alloc(LAYER3, LAYER2);

    int i, j;
    for (i = 0; i < LAYER2; i++)
        h1_biases[i] = (float) rand() / RAND_MAX * 6 - 3;
    for (i = 0; i < LAYER2; i++)
        h2_biases[i] = (float) rand() / RAND_MAX * 6 - 3;
    for (i = 0; i < LAYER3; ++i)
        o_biases[i] = (float) rand() / RAND_MAX * 6 - 3;
    for (i = 0; i < LAYER2; ++i)
        for (j = 0; j < LAYER1; ++j)
            h1_weights[i][j] = (float) rand() / RAND_MAX * 6 - 3;
    for (i = 0; i < LAYER2; ++i)
        for (j = 0; j < LAYER2; ++j)
            h2_weights[i][j] = (float) rand() / RAND_MAX * 6 - 3;
    for (i = 0; i < LAYER3; ++i)
        for (j = 0; j < LAYER2; ++j)
            o_weights[i][j] = (float) rand() / RAND_MAX * 6 - 3;

    NEURALNETWORK network;

    network.h1_biases = h1_biases;
    network.h2_biases = h2_biases;
    network.o_biases = o_biases;

    network.h1_weights = h1_weights;
    network.h2_weights = h2_biases;
    network.o_weights = o_weights;

    dynamic_array_free(h1_weights, LAYER2);
    dynamic_array_free(h2_weights, LAYER2);
    dynamic_array_free(o_weights, LAYER3);
    return network;
}

int forward_propagation( NEURALNETWORK *network, double *x ) {
    double z1[LAYER2];
    double z2[LAYER2];
    double o[LAYER3];
    int i;

    mxv(network->h1_weights, x, z1, LAYER2, LAYER1);
    plus(z1, network->h1_biases, z1, LAYER2);
    for (i = 0; i < LAYER2; i++)
        z1[i] = sigmoid(z1[i]);

    mxv(network->h2_weights, z1, z2, LAYER2, LAYER2);
    plus(z2, network->h2_biases, z2, LAYER2);
    for (i = 0; i < LAYER2; i++)
        z2[i] = sigmoid(z2[i]);

    mxv(network->o_weights, z1, o, LAYER3, LAYER2);
    plus(o, network->o_biases, o, LAYER3);
    for (i = 0; i < LAYER3; i++)
        o[i] = sigmoid(o[i]);

    return max_ind(o, LAYER3);
}

void update_batch( NEURALNETWORK *network, double *batch, int l_rate )
{

}

double backprop( NEURALNETWORK *network, double *x, double *y )
{
    return 0;
}

void fit( NEURALNETWORK *network, double data, int epochs, int mini_batch_size, int l_rate )
{

}