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
Соответственно, каждый из 10 выходов формируется как линейная комбинация 784 входов. Таким образом, модель имеет 784⋅10=7840 параметров, которые надо натренировать.
___
