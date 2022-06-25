#ifndef NNET_NEURALNETWORK_H
#define NNET_NEURALNETWORK_H

typedef struct {

    double h1_biases[100];
    double h2_biases[100];
    double o_biases[10];

    double h1_weights[100][784];
    double h2_weights[100][100];
    double o_weights[10][100];
} NEURALNETWORK;

#endif
