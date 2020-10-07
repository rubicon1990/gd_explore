import tensorflow as tf
from tensorflow import keras
import math
import numpy as np
import sys
import matplotlib.pyplot as plt

"""
y = mean(x_i * a^2) + 10 * a
  = mean(x_i) * a^2 + 10 * a
min when 2 * a * mean(x_i) == -10
thus, argmin a = -5 / mean

1. batch update
    grad = 2 * a * mean(xi)
    updated_a = a - learning_rate * grad

2. batch update, distributed version
    total_grad = sum(grad_i * n_i) / sum(n_i)
    updated_a = a - lr * total_grad
              = a - sum(lr * grad_i * n_i) / sum(n_i)
              
    Q: Does this always hold true? 
    A: Depend on the relation between total_grad and grad_i.

3. mini-batch update, distributed version?

"""


def loss_calulate(x, a0):
    return x.mean() * a0 ** 2 + 10 * a0


def argmin(x):
    return -5 / x.mean()


def train_step(x, a, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(a)
        l = loss_calulate(x, a)
    grads = tape.gradient(l, a)
    return a - learning_rate * grads


def plotting(history_loss, triple_loss_func):
    _a_min, _a_s, _l_s = triple_loss_func

    plt.subplot(1, 3, 1)
    plt.plot(history_loss['l'], label='loss', color='black')
    plt.title('loss history')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend(loc='best')

    plt.subplot(1, 3, 2)
    plt.plot(history_loss['a'], label='a', color='black')
    plt.title('param history')
    plt.xlabel('iteration')
    plt.ylabel('a')
    plt.legend(loc='best')

    plt.subplot(1, 3, 3)
    plt.plot(_a_s, _l_s, label='loss', color='black')
    plt.axvline(_a_min, label='a_min = %.4f' % _a_min, color='r')
    plt.title('loss function')
    plt.xlabel('a')
    plt.ylabel('loss')
    plt.legend(loc='best')


if __name__ == "__main__":
    x_train = np.arange(10000)
    a = tf.constant(30.0)
    learning_rate = 0.0001
    n_iter = 100
    
    print(0, a.numpy(), loss_calulate(x_train, a).numpy())
    hist_loss = {'l': [loss_calulate(x_train, a).numpy()],
                 'a': [a.numpy()]}
    
    for i in range(1, n_iter + 1):
        a = train_step(x_train, a, learning_rate)
        loss = loss_calulate(x_train, a).numpy()
        print(i, a.numpy(), loss)
        hist_loss['l'].append(loss)
        hist_loss['a'].append(a)
    
    a_min = argmin(x_train)
    a_s = np.linspace(a_min - 1, a_min + 1, 100)
    l_s = (lambda _a: loss_calulate(x_train, _a))(a_s)
    
    plt.figure(figsize=(15, 5))
    plotting(hist_loss, (a_min, a_s, l_s))
    plt.tight_layout()
    plt.show()
