#! /usr/bin/env python3

# Туториал из урока https://www.youtube.com/watch?v=HA-F6cZPvrg

import numpy as np
import sys


# Функция активации нейрона
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mse(old_val, new_val):
    return np.mean((old_val-new_val)**2)


class PartyNN(object):
    def __init__(self, learning_rate=0.1):
        # Случайные веса перехода из входа на 1й слой, нормальное распределение (колокол),
        # центр распределения в 0.0, минимум и максимум - (2 ** -0.5 = 0.7071067811865476)?
        # 3 входа, 2 выхода
        self.weights_0_1 = np.random.normal(0.0, 2 ** -0.5, (2, 3))
        # Случайные веса перехода из 1го слоя на выход, нормальное распределение (колокол),
        # центр распределения в 0.0, минимум и максимум - (-1,1)?
        # 2 входа, 1 выход
        self.weights_1_2 = np.random.normal(0.0, 1, (1, 2))
        # Обертка, котора применяет функцию sigmoid к вектору значений
        self.sigmoid_mapper = np.vectorize(sigmoid)
        # Скорость обучения
        self.learning_rate = np.array([learning_rate])

    # Метод прямого прохода нейронной сети
    def predict(self, inputs):
        # Выполняем скалярное перемножение весов первого слоя и входных значений
        inputs_1 = np.dot(self.weights_0_1, inputs)
        # Получаем значения пороговой функции для каждого значения, полученного из весов,
        # результат - это значение выхода из первого слоя
        outputs_1 = self.sigmoid_mapper(inputs_1)

        # Выполняем скалярное перемножение весов второго слоя и выходных значений первого слоя
        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        # Получаем значения пороговых функций для каждого значения, полученного из весов -
        # результат - это значение выхода из второго слоя
        outputs_2 = self.sigmoid_mapper(inputs_2)

        return outputs_2

    # Метод тренировки нейронной сети
    def train(self, inputs, expected_predict):     
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)
        
        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.sigmoid_mapper(inputs_2)
        actual_predict = outputs_2[0]
        
        error_layer_2 = np.array([actual_predict - expected_predict])
        gradient_layer_2 = actual_predict * (1 - actual_predict)
        weights_delta_layer_2 = error_layer_2 * gradient_layer_2  
        self.weights_1_2 -= (np.dot(weights_delta_layer_2, outputs_1.reshape(1, len(outputs_1)))) * self.learning_rate
        
        error_layer_1 = weights_delta_layer_2 * self.weights_1_2
        gradient_layer_1 = outputs_1 * (1 - outputs_1)
        weights_delta_layer_1 = error_layer_1 * gradient_layer_1
        self.weights_0_1 -= np.dot(inputs.reshape(len(inputs), 1), weights_delta_layer_1).T  * self.learning_rate


def main():
    # Массив с правильными значениями на которых надо тренироваться
    train = [
        ([0, 0, 0], 0),
        ([0, 0, 1], 1),
        ([0, 1, 0], 0),
        ([0, 1, 1], 0),
        ([1, 0, 0], 1),
        ([1, 0, 1], 1),
        ([1, 1, 0], 0),
        ([1, 1, 1], 0),
    ]

    # epochs = 5000
    # learning_rate = 0.05
    epochs = 50000
    learning_rate = 0.2

    # Создаем сеть
    network = PartyNN(learning_rate=learning_rate)

    # Количество итераций обучения
    for e in range(epochs):
        inputs_ = []
        correct_predictions = []

        # Перебираем массив верных значений и результатов для обучения
        for input_stat, correct_predict in train:
            input_array = np.array(input_stat)

            # Тренируем сеть
            network.train(input_array, correct_predict)

            # Сохраняем верный вход и верный выход для прямого прохода
            inputs_.append(input_array)
            correct_predictions.append(np.array(correct_predict))

        if e % (epochs/10) == 0:
            # Выполняем прямой проход
            current_result = network.predict(np.array(inputs_).T)

            # Сравниваем текущий результат и верный результат квадратичной функцией
            train_loss = mse(current_result, np.array(correct_predictions))

            # Выводим текущую ошибку
            progress = str(100 * e / float(epochs))[:4]
            sys.stdout.write("Progress: {}, training loss: {}\n".format(progress, str(train_loss)))

    print("\n")
    for input_stat, correct_predict in train:
        print("For input: {} the prediction is: {}, expected: {}".format(
            str(input_stat),
            str(network.predict(np.array(input_stat))),
            str(correct_predict)))

    # Выведем веса нашей обученной сети
    print("\n")
    print("Layer 0->1 weights:\n{}\n".format(str(network.weights_0_1)))
    print("Layer 1->2 weights:\n{}\n".format(str(network.weights_1_2)))


if __name__ == '__main__':
    main()
