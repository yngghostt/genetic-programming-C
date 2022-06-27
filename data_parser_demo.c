#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATASET_SIZE 5

struct result{
  int **layers;
  int **answers;
};

int **dynamic_array_alloc(size_t N, size_t M)
{
    int **A = (int **)malloc(N*sizeof(int *));
    int i;
    for(i = 0; i < N; i++) {
        A[i] = (int *)malloc(M*sizeof(int ));
    }
    return A;
}

void dynamic_array_free(int **A, size_t N)
{
    int i;
    for(i = 0; i < N; i++) {
        free(A[i]);
    }
    free(A);
}

void picture_csv_parser(int **layers, int **answers){
  int i, j;
  int counter = 0;
  FILE *dataset;
  char picture[10000] = {0};
  if((dataset = fopen("dataset.txt", "r")) == NULL){
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
        layers[i][j] = atoi(token);
        token = strtok_r(NULL, " ,.-", &last);
    }
    token = strtok_r(NULL, " ,.-", &last);
    answers[i][atoi(token)] = 1;
  }
}

int main(void) {
  int i, j;
  int **layers = dynamic_array_alloc(DATASET_SIZE, 784);
  int **answers = dynamic_array_alloc(DATASET_SIZE, 10);
  for(i = 0; i < DATASET_SIZE; i++){
    for(j = 0; j < 785; j++){
      layers[i][j] = 0;
    }
  }
  for(i = 0; i < DATASET_SIZE; i++){
    for(j = 0; j < 10; j++){
      answers[i][j] = 0;
    }
  }

  picture_csv_parser(layers, answers);
  for(i = 0; i < 783; i++){printf("%i", layers[2][i]);}

}
