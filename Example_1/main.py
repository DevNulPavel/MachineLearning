#! /usr/bin/env python3

# Туториал из урока https://www.youtube.com/watch?v=AZG0j0pNY-4

import numpy as np

vodka = 0.0
rain = 0.0
friend = 1.0


# Функция активации нейрона
def activation_function(x):
    if x >= 0.5:
        return 1
    else:
        return 0


# Выполняем нашу нейронную сеть
def predict(in_vodka, in_rain, in_friend):
    # Входной массив условий
    input_array = [in_vodka, in_rain, in_friend]
    inputs = np.array(input_array)
    print("Input: " + str(input_array))

    # Веса срытых слоев
    weights_input_to_hiden_1 = [0.25, 0.25, 0]
    weights_input_to_hiden_2 = [0.5, -0.4, 0.9]
    weight_hidden_to_output = [-1, 1]

    # Веса переходов между слоями
    weights_input_to_hiden = np.array([weights_input_to_hiden_1, weights_input_to_hiden_2])
    weights_hidden_to_output = np.array(weight_hidden_to_output)

    # Скалярное произведение векторов входа и первого слоя
    hiden_input = np.dot(weights_input_to_hiden, inputs)
    print("Hidden input: " + str(hiden_input))

    # Выход из первого скрытого слоя
    hiden_output = np.array([activation_function(x) for x in hiden_input])
    print("Hidden output: " + str(hiden_output))

    # Скалярное произведение выхода первого слоя и весов первого слоя
    output = np.dot(weights_hidden_to_output, hiden_output)
    # print("output: " + str(output))

    return activation_function(output) == 1


def main():
    print("Result: " + str(predict(vodka, rain, friend)))


if __name__ == '__main__':
    main()
