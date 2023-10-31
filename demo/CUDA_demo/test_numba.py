from numba import njit
import random
import time

@njit
def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

for i in range(10):
    monte_carlo_pi(10000)
t1 = time.time()
monte_carlo_pi(10000)
t2 = time.time()
print('Time: {} ms'.format((t2-t1)*1000))