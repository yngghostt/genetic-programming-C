#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data_parcer.h"

void picture_csv_parser(double **layers, double **answers){
    int i, j;
    FILE *dataset;
    char picture[10000] = {0};
    if((dataset = fopen("D:\\projects\\CLionProjects\\nnet\\mnist_784.txt", "r")) == NULL){
        printf("Impossible to open this file.");
        return;
    }
    fgets(picture, 10000, dataset);
    for(i = 0; i < DATASET_SIZE; i++){
        if(fgets(picture, 10000, dataset) == NULL){
            printf("End of dataset.");
            return;
        }

        char *token, *last;
        token = strtok_r(picture, " ,", &last);
        for(j = 0; j < 783; j++) {
            layers[i][j] = (double)atoi(token);
            token = strtok_r(NULL, " ,.-", &last);
        }
        token = strtok_r(NULL, " ,.-", &last);
        answers[i][atoi(token)] = 1;
    }
}
