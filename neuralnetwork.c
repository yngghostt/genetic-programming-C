#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "NEURALNETWORK.h"

double sigmoid( double x )
{
    return 1/(1+ exp(-1*x));
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
