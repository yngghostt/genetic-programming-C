#include <math.h>
#include <stdio.h>
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
    NEURALNETWORK network;

    network.h1_biases = (double *)malloc(LAYER2*sizeof(double));
    network.h2_biases = (double *)malloc(LAYER2*sizeof(double));
    network.o_biases = (double *)malloc(LAYER3*sizeof(double));

    network.h1_weights = dynamic_array_alloc(LAYER2, LAYER1);
    network.h2_weights = dynamic_array_alloc(LAYER2, LAYER2);
    network.o_weights = dynamic_array_alloc(LAYER3, LAYER2);

    int i, j;
    for (i = 0; i < LAYER2; i++)
        network.h1_biases[i] = (((double)rand())/RAND_MAX)*4 - 2;

    for (i = 0; i < LAYER2; i++)
        network.h2_biases[i] = (((double)rand())/RAND_MAX)*4 - 2;

    for (i = 0; i < LAYER3; i++)
        network.o_biases[i] = (((double)rand())/RAND_MAX)*4 - 2;

    for (i = 0; i < LAYER2; i++)
        for (j = 0; j < LAYER1; j++)
            network.h1_weights[i][j] = (((double)rand())/RAND_MAX)*4 - 2;

    for (i = 0; i < LAYER2; i++)
        for (j = 0; j < LAYER2; j++)
            network.h2_weights[i][j] = (((double)rand())/RAND_MAX)*4 - 2;

    for (i = 0; i < LAYER3; i++)
        for (j = 0; j < LAYER2; j++)
            network.o_weights[i][j] = (((double)rand())/RAND_MAX)*4 - 2;

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

void back_propagation( NEURALNETWORK *network, DELTA *delta, double *x, double *y )
{
    double z1[LAYER2];
    double a1[LAYER2];

    double z2[LAYER2];
    double a2[LAYER2];

    double z_o[LAYER3];
    double predicted[LAYER3];

    double delta1[LAYER3];
    double delta2[LAYER2];
    double delta3[LAYER2];

    double tmp[LAYER3] = {0};

    double **o_del_w = dynamic_array_alloc(LAYER3, LAYER2);
    double **h2_del_w = dynamic_array_alloc(LAYER2, LAYER2);
    double **h1_del_w = dynamic_array_alloc(LAYER2, LAYER1);

    int i;

    //Прямой проход
    mxv(network->h1_weights, x, z1, LAYER2, LAYER1);
    plus(z1, network->h1_biases, z1, LAYER2);
    for (i = 0; i < LAYER2; i++)
        a1[i] = sigmoid(z1[i]);

    mxv(network->h2_weights, z1, z2, LAYER2, LAYER2);
    plus(z2, network->h2_biases, z2, LAYER2);
    for (i = 0; i < LAYER2; i++)
        a2[i] = sigmoid(z2[i]);

    mxv(network->o_weights, z1, z_o, LAYER3, LAYER2);
    plus(z_o, network->o_biases, z_o, LAYER3);
    for (i = 0; i < LAYER3; i++)
        predicted[i] = sigmoid(z_o[i]);

    //Обратный проход
    minus(predicted, y, delta1, LAYER3);
    for(i = 0; i < LAYER3; i++)
    {
        delta1[i] *= dsigmoid(z_o[i]);
        tmp[i] = dsigmoid(z_o[i]);
    }

    vxv(delta1, a2, o_del_w,LAYER3, LAYER2);

    double **o_weights_tr = transpose(network->o_weights, LAYER3, LAYER2);
    mxv(o_weights_tr, delta1, delta2, LAYER2, LAYER3);
    for(i = 0; i < LAYER2; i++)
        delta2[i] *= dsigmoid(z2[i]);
    vxv(delta2, a1, h2_del_w,LAYER2, LAYER2);

    double **h2_weights_tr = transpose(network->h2_weights, LAYER2, LAYER2);
    mxv(h2_weights_tr, delta2, delta3, LAYER2, LAYER2);
    for(i = 0; i < LAYER2; i++)
        delta3[i] *= dsigmoid(z1[i]);
    vxv(delta1, x, h1_del_w,LAYER2, LAYER1);

    dynamic_array_free(o_weights_tr, LAYER2);
    dynamic_array_free(h2_weights_tr, LAYER2);

    delta->o_del_b = delta1;
    delta->h2_del_b = delta2;
    delta->h1_del_b = delta3;

    delta->o_del_w = o_del_w;
    delta->h2_del_w = h2_del_w;
    delta->h1_del_w = h1_del_w;
}

void update_batch( NEURALNETWORK *network, double *batch, int l_rate )
{

}

void fit( NEURALNETWORK *network, double data, int epochs, int mini_batch_size, int l_rate )
{

}
