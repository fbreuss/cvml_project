from numba import jit
import numpy as np
import random
# to measure exec time
from timeit import default_timer as timer


def func(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

@jit(nopython=True)
def func2(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

@jit(nopython=True)
def func3(nsamples):
    return func2(nsamples)

if __name__ == "__main__":
    n = 10000000

    start = timer()
    func(n)
    print("without GPU:", timer() - start)

    start = timer()
    func3(n)
    print("with GPU:", timer() - start)
