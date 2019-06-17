#! /usr/bin/env python3
# coding=utf-8

# Туториал из урока
# https://github.com/dataPAPA/tensorflow_mnist/blob/master/mnist.py
# https://www.youtube.com/watch?v=R_Lmewg8W64
# https://github.com/tensorflow/tensorflow
# https://www.tensorflow.org/tutorials/



import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def main():
    # Загружаем MNIST датасет - числа, написанные от руки
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    # Задаем граф вычислений в тензорфлоу
    # Плейсхолдеры - те места, куда будут подставляться значения входных-выходных переменных
    # Входные значения, картинка 28x28
    x = tf.placeholder("float", [None, 784])
    # Выходные значения 10 чисел
    y = tf.placeholder("float", [None, 10])

    # Переменные - это веса нашего единственного слоя сети
    # Формула W*x + b
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Предсказание - это линейное преобразование входного вектора.
    # До преобразования размерность - 784
    # После преобразования - 10
    linear_prediction = tf.matmul(x, W) + b
    scaled_prediction = tf.nn.softmax(linear_prediction) # Softmax

    # Функция потерь - кросс энтропия. В двух словах не объясню почему так.
    # Почитайте лучше википедию. Но она работает
    loss_function = tf.losses.softmax_cross_entropy(y, linear_prediction)

    # Оптимизатор - у нас простой градиентный спуск
    learning_rate = 0.025
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

    # Инициализируем сессию, с которой будем работать
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    # Цикл обучения. Учимся на минибатчах, каждые 5 шагов печатаем ошибку
    batch_size = 250

    for iteration in range(100000):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if iteration % 5000 == 0:
            loss = loss_function.eval({x: mnist.test.images, y: mnist.test.labels})
            print ("#{}, loss={:.4f}".format(iteration, loss))

    # Задаем граф вычислений, выдающий точность предсказания
    predicted_label = tf.argmax(scaled_prediction, 1)
    actual_label = tf.argmax(y, 1)
    is_equal_labels = tf.equal(actual_label, predicted_label)
    accuracy = tf.reduce_mean(tf.cast(is_equal_labels, "float"))

    # Вычисляем точность
    accracy_value = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
    print ("Accuracy:", accracy_value)

    # Предсказываем лейбы для тествого датасета
    predicted_label = tf.argmax(scaled_prediction, 1)
    predicted_test_values = predicted_label.eval({x: mnist.test.images})
    print ("Predictions: {}".format(predicted_test_values))


if __name__ == '__main__':
    main()
