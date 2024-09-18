from cma import CMA
import math as m
import tensorflow as tf
import numpy as np

def xsy4(x):
    N, D = x.shape
    sines = tf.zeros([N])
    sinesqrts = tf.zeros([N])
    squares = tf.zeros([N])
    for d in range(D):
        xd = x[:, d]
        sines += tf.sin(xd) ** 2
        squares += xd ** 2
        sinesqrts += tf.sin(tf.sqrt(tf.sqrt(xd**2)))**2

    return (sines-tf.exp(-squares)) * tf.exp(-sinesqrts)


def salomon(x):
    norm = tf.linalg.norm(x, axis=-1)
    return 1 + 0.1 * norm - tf.cos(2*m.pi*norm)


def griewank(x):
    N, D = x.shape
    s = tf.ones([N])
    summands = tf.zeros([N])
    factors = tf.ones([N])
    for d in range(D):
        xd = x[:, d]
        summands += xd * xd
        factors *= tf.cos(xd/((d+1)**.5))
    summands /= 4000
    return s + summands + factors

def unknown(x):
    norm = tf.linalg.norm(x, axis=-1)
    return 1 - tf.cos(2*m.pi*norm)+0.1*norm

def rastrigin(x):
    N, D = x.shape
    s = tf.zeros([N])
    A = 10
    for d in range(D):
        xd = x[:, d]
        s += xd*xd - A*tf.cos(2*m.pi*xd)+10
    return A*D + s


def ackley(x):
    N, D = x.shape
    cs = tf.zeros([N])
    ns = tf.zeros([N])
    for d in range(D):
        xd = x[:, d]
        cs += tf.cos(2*m.pi*xd)
        ns += xd*xd
    cs /= D
    ns = tf.sqrt(ns)
    ns *= -.2/m.sqrt(D)
    return -20*tf.exp(ns)-tf.exp(cs) + 20 + m.e


x0 = [85. for _ in range(50)]

f = xsy4

N_RUNS = 5
res = []

for _ in range(N_RUNS):
    optim = CMA(
        initial_solution=x0,
        initial_step_size=1.0,
        fitness_function=f,
        population_size=100
    )

    best_solution, best_fitness = optim.search()
    print(best_fitness)
    res += [best_fitness]

res = np.array(res)
print("Mean: ", res.mean())
print("STD: ", res.std())


