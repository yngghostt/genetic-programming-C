# Реализация нейронной сети для решения задачи MNIST 784.
---
## Что такое нейронная сеть?

Нейронная сеть - это метод в искусственном интеллекте, который учит компьютеры обрабатывать данные таким же способом, как и человеческий мозг. Это тип процесса машинного обучения, называемый глубоким обучением, который использует взаимосвязанные узлы или нейроны в слоистой структуре, напоминающей человеческий мозг. Он создает адаптивную систему, с помощью которой компьютеры учатся на своих ошибках и постоянно совершенствуются.
___
## Как работают нейронный сети?

Архитектура нейронных сетей повторяет структуру человеческого мозга. Клетки человеческого мозга, называемые нейронами, образуют сложную сеть с высокой степенью взаимосвязи и посылают друг другу электрические сигналы, помогая людям обрабатывать информацию. Точно так же искусственная нейронная сеть состоит из искусственных нейронов, которые взаимодействуют для решения проблем. Искусственные нейроны — это программные модули, называемые узлами, а искусственные нейронные сети — это программы или алгоритмы, которые используют вычислительные системы для выполнения математических вычислений.
___
## MNIST 784

MNIST - это один из классических датасетов на котором принято пробовать всевозможные подходы к классификации изображений. Набор содержит черно-белые изображения размера 28×28 пикселей рукописных цифр от 0 до 9.
На входе 784=28⋅28 нейрона, каждый подключен к одному из пикселей изображения. На выходе слой с 10-ю нейронами по одному на цифру.
![Иллюстрация к проекту](https://github.com/yngghostt/neural-network-C/blob/main/neural.png)
___

## Реализация нейронной сети

### Структура нейронной сети

Сеть состоит из 4х слоёв: входной, 2 скрытых слоя, выходной. Размеры слоёв задаются константами. По умнолчанию, размер входного слоя: 784 (нейрон для каждого пикселя), размер выходного слоя: 10 (нейрон для каждой цифры). Для каждого слоя хранится одномерный массив из смещений и двумерный массив весов.

Нейронная сеть представлена структурой, в которой хранятся ссылки на массивы с параметрами нейронной сети.
```с
typedef struct {
    double *h1_biases;
    double *h2_biases;
    double *o_biases;

    double **h1_weights;
    double **h2_weights;
    double **o_weights;
} NEURALNETWORK;
```

### Функции нейронной сети

1. Инициализация. Изначально все веса в слоях задаются случайно, чтобы после этого они улучшались. Функция NEURALNETWORK init( void ) создает нейронную сеть и заполняет ее слои случайными числами. 

2. Прямое распространение. Во время работы функции int forward_propagation( NEURALNETWORK *network, int *x ) выходные данные функции активации в одном слое будут передаваться в качестве входных данных на следующий уровень, пока не будут получены данные в выходном слое. 

3. Обратное распространение. При работе функции void back_propagation( NEURALNETWORK *network, DELTA *delta, int *x, int *y ) вычисляется ошибка (вектор, показывающий разницу между желаемым результатом и тем, что выдала сеть). Затем эта ошибка передается в обратном направлении от выходного слоя к скрытым, где происходит корректировка весов. 

4. Функция void update_batch( NEURALNETWORK *network, int **batch_x, int **batch_y, int start, int size, double l_rate ) выполняет прямое и обратное распространение после прохода по некоторому небольшому пакету картинок. Это сделано для большего охвата данных - невозможно загрузить в память весь датасет, так как он содержит в себе 70000 строк. 

5. Наконец, функция void fit( NEURALNETWORK *network, int **data_x, int **data_y, long int data_size, int epochs, int mini_batch_size, double l_rate ) обучает сеть. Она разбивает данные из датасета на пакеты и вызывает для каждого из них update_batch. 

6. Необходимо не только обучить сеть, но и оценить, насколько она точна. Точность измеряется множеством проходов на тестовых данных и оценивает, насколько прогнозы сети отличаются от реальных. Данный алгоритм реализован в функции double accuracy(NEURALNETWORK *network, int **test_x, int **test_y, int start, int test_size )

### Дополнительные функции

Для полной реализации нейросети было необходимо реализовать пакет векторно-матричной математики, а также некоторые вспомогательные функции. 
```c
extern double **dynamic_array_alloc( size_t N, size_t M );
extern int **int_dynamic_array_alloc(size_t N, size_t M);
extern double **dynamic_array_alloc_zeros(size_t N, size_t M);
extern void dynamic_array_free( double **A, size_t N );
extern void print( double *x, int n );
void matrix_print( double **x, int rows, int cols );
extern double dot( double *v, double *y, int n );
extern double norm( double *x, int n );
extern void mxv(  double **m,  double *v,double *res, int rows, int cols );
extern void vxv( double *x, double *y, double **res, int n, int m );
extern void plus( double *x, double *y, double *res, int n );
extern void minus( double *x, double *y, double *res, int n );
extern void multiply( double *x, double a, double *res, int n );
extern void matrix_plus( double **a, double **b, double **res, int rows, int cols );
extern void matrix_minus( double **a, double **b, double **res, int rows, int cols );
extern void matrix_multiply( double **m, double a, double **res, int rows, int cols );
extern double **transpose( double **m, int rows, int columns );
extern int max_ind( double *x, int n );
```

Пакет содержит функции для удобной работы с динамическими массивами данных, а также основные векторно-матричные действия (сложение, вычитание, различные умножения, транспозиция и поиск максимального индекса). 

Кроме того, был реализован парсер данных. Картинки в датасете хранятся построчно, каждое значение пикселя записано через запятую. Данный формат необходимо было превратить в массив массивов, содержащих целые числа. Реализация представлена в функции void picture_csv_parser( double **layers, double **answers ).

### Запуск обучения

```c
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "data_parcer.h"
#include "NEURALNETWORK.h"
#include "vectormath.h"

int main() {
    srand(time(NULL));

    int **layers = int_dynamic_array_alloc(DATASET_SIZE, 784);
    int **answers = int_dynamic_array_alloc_zeros(DATASET_SIZE, 10);

    picture_csv_parser(layers, answers);

    NEURALNETWORK neuralnetwork = init();
    fit(&neuralnetwork, layers, answers, DATASET_SIZE, 5, 100, 5);
    double acc = accuracy(&neuralnetwork, layers, answers, DATASET_SIZE, 100);
    printf("%f", acc);
    getchar();
}
```
