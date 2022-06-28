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
        network.h1_biases[i] = (((double)rand())/RAND_MAX)*2 - 1;

    for (i = 0; i < LAYER2; i++)
        network.h2_biases[i] = (((double)rand())/RAND_MAX)*2 - 1;

    for (i = 0; i < LAYER3; i++)
        network.o_biases[i] = (((double)rand())/RAND_MAX)*2 - 1;

    for (i = 0; i < LAYER2; i++)
        for (j = 0; j < LAYER1; j++)
            network.h1_weights[i][j] = (((double)rand())/RAND_MAX)*2 - 1;

    for (i = 0; i < LAYER2; i++)
        for (j = 0; j < LAYER2; j++)
            network.h2_weights[i][j] = (((double)rand())/RAND_MAX)*2 - 1;

    for (i = 0; i < LAYER3; i++)
        for (j = 0; j < LAYER2; j++)
            network.o_weights[i][j] = (((double)rand())/RAND_MAX)*2 - 1;

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

    mxv(network->o_weights, z2, o, LAYER3, LAYER2);
    plus(o, network->o_biases, o, LAYER3);
    for (i = 0; i < LAYER3; i++)
        o[i] = sigmoid(o[i]);

    return max_ind(o, LAYER3);
}

void back_propagation( NEURALNETWORK *network, DELTA *delta, double *x, double *y )
{
    double *z1 = (double *)malloc(LAYER2*sizeof(double));
    double *a1 = (double *)malloc(LAYER2*sizeof(double));
    double *z2 = (double *)malloc(LAYER2*sizeof(double));
    double *a2 = (double *)malloc(LAYER2*sizeof(double));
    double *z_o = (double *)malloc(LAYER3*sizeof(double));
    double *predicted = (double *)malloc(LAYER3*sizeof(double));

    double *delta1 = (double *)malloc(LAYER3*sizeof(double));
    double *delta2 = (double *)malloc(LAYER2*sizeof(double));
    double *delta3 = (double *)malloc(LAYER2*sizeof(double));

    double **o_del_w = dynamic_array_alloc(LAYER3, LAYER2);
    double **h2_del_w = dynamic_array_alloc(LAYER2, LAYER2);
    double **h1_del_w = dynamic_array_alloc(LAYER2, LAYER1);

    int i;

    //Прямой проход
    mxv(network->h1_weights, x, z1, LAYER2, LAYER1);
    plus(z1, network->h1_biases, z1, LAYER2);
    for (i = 0; i < LAYER2; i++)
        a1[i] = sigmoid(z1[i]);

    mxv(network->h2_weights, a1, z2, LAYER2, LAYER2);
    plus(z2, network->h2_biases, z2, LAYER2);
    for (i = 0; i < LAYER2; i++)
        a2[i] = sigmoid(z2[i]);

    mxv(network->o_weights, a2, z_o, LAYER3, LAYER2);
    plus(z_o, network->o_biases, z_o, LAYER3);
    for (i = 0; i < LAYER3; i++)
        predicted[i] = sigmoid(z_o[i]);

    //Обратный проход
    minus(predicted, y, delta1, LAYER3);
    for(i = 0; i < LAYER3; i++)
    {
        delta1[i] *= dsigmoid(z_o[i]);
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

    free(a1);
    free(z1);
    free(a2);
    free(z2);
    free(predicted);
    free(z_o);

}

void update_batch( NEURALNETWORK *network, double **batch_x, double **batch_y, int start, int size, double l_rate )
{
    DELTA delta;

    double *o_b = (double *)malloc(LAYER3*sizeof(double));
    double *h2_b = (double *)malloc(LAYER2*sizeof(double));
    double *h1_b = (double *)malloc(LAYER2*sizeof(double));

    double **o_w = dynamic_array_alloc_zeros(LAYER3, LAYER2);
    double **h2_w = dynamic_array_alloc_zeros(LAYER2, LAYER2);
    double **h1_w = dynamic_array_alloc_zeros(LAYER2, LAYER1);


    int i;
    for (i = 0; i < LAYER3; i++)
        o_b[i] = 0;
    for (i = 0; i < LAYER2; i++)
        h2_b[i] = 0;
    for (i = 0; i < LAYER2; i++)
        h1_b[i] = 0;

    for(i = start; i < start + size; i++)
    {
        back_propagation(network, &delta, batch_x[i], batch_y[i]);

        plus(o_b, delta.o_del_b, o_b, LAYER3);
        plus(h2_b, delta.h2_del_b, h2_b, LAYER2);
        plus(h1_b, delta.h1_del_b, h1_b, LAYER2);

        matrix_plus(o_w, delta.o_del_w, o_w, LAYER3, LAYER2);
        matrix_plus(h2_w, delta.h2_del_w, h2_w, LAYER2, LAYER2);
        matrix_plus(h1_w, delta.h1_del_w, h1_w, LAYER2, LAYER1);
    }

    double rate = l_rate/size;

    multiply(o_b, rate, o_b, LAYER3);
    minus(network->o_biases, o_b, network->o_biases, LAYER3);

    multiply(h2_b, rate, h2_b, LAYER2);
    minus(network->h2_biases, h2_b, network->h2_biases, LAYER2);

    multiply(h1_b, rate, h1_b, LAYER2);
    minus(network->h1_biases, h1_b, network->h1_biases, LAYER2);

    matrix_multiply(o_w, rate, o_w, LAYER3, LAYER2);
    matrix_minus(network->o_weights, o_w, network->o_weights, LAYER3, LAYER2);

    matrix_multiply(h2_w, rate, h2_w, LAYER2, LAYER2);
    matrix_minus(network->h2_weights, h2_w, network->h2_weights, LAYER2, LAYER2);

    matrix_multiply(h1_w, rate, h1_w, LAYER2, LAYER1);
    matrix_minus(network->h1_weights, h1_w, network->h1_weights, LAYER2, LAYER1);

    free(o_b);
    free(h2_b);
    free(h1_b);

    dynamic_array_free(o_w, LAYER3);
    dynamic_array_free(h2_w, LAYER3);
    dynamic_array_free(h1_w, LAYER3);
}

void fit( NEURALNETWORK *network, double **data_x, double **data_y, long int data_size, int epochs, int mini_batch_size, double l_rate )
{
    int i, j;
    for(i = 0; i < epochs; i++)
    {
        for(j = 0; j < data_size; j += mini_batch_size)
        {
            update_batch(network, data_x, data_y, j,mini_batch_size, l_rate);

        }
        printf("Epoch %d passed\n", i+1);
    }
}

double accuracy(NEURALNETWORK *network, double **test_x, double **test_y, int start, int test_size )
{
    int i, pred, ans, score = 0;
    for(i = start; i < start + test_size; i++)
    {
        pred = forward_propagation(network, test_x[i]);
        ans = max_ind(test_y[i], LAYER3);
        if (pred == ans)
            score++;
    }

    return (double)score/test_size;
}
