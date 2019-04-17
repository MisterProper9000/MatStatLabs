import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import laplace
from scipy.stats import uniform
from scipy.stats import poisson


POISSON_PARAM = 3
UNIFORM_FRONT = 4


def normalized_distribution(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-x * x / 2)


def laplace_distribution(x):
    return (1 / np.sqrt(2)) * np.exp(-np.sqrt(2) * np.abs(x))


def uniform_distribution(x):
    flag = (x <= UNIFORM_FRONT)
    return 1 / (2 * UNIFORM_FRONT) * flag


def cauchy_distribution(x):
    return 1 / (np.pi * (1 + x * x))


def poisson_distribution(x):
    n = x.size
    res = []
    for i in range(n):
        res.append(0)

    for i in range(n):
        res[i] = (1 / np.power(np.e, 3)) / np.math.factorial(int(x[i])) * np.power(3, int(x[i]))
    return res


func_density_dict = {
    'normal':   normalized_distribution,
    'laplace':  laplace_distribution,
    'uniform':  uniform_distribution,
    'cauchy':   cauchy_distribution,
    'poisson':  poisson_distribution,
}


def cumulative_laplace(x):
    return laplace.cdf(x, 0, 1/np.sqrt(3))


def cumulative_poisson(x):
    return poisson.cdf(x, POISSON_PARAM)


def cumulative_cauchy(x):
    return (1/np.pi) * np.arctan(x) + 0.5


def cumulative_uniform(x):
    return uniform.cdf(x, -4, 8)


func_cumulative_dict = {
    'normal': norm.cdf,
    'laplace': cumulative_laplace,
    'uniform': cumulative_uniform,
    'cauchy': cumulative_cauchy,
    'poisson': cumulative_poisson
}


def generate_laplace(x):
    return np.random.laplace(0, 1/np.sqrt(3), x)


def generate_uniform(x):
    return np.random.uniform(-UNIFORM_FRONT, UNIFORM_FRONT, x)


def generate_poisson(x):
    return np.random.poisson(POISSON_PARAM, x)


generate_dict = {
    'normal':   np.random.standard_normal,
    'laplace':  generate_laplace,
    'uniform':  generate_uniform,
    'cauchy':   np.random.standard_cauchy,
    'poisson':  generate_poisson,
}


def empirical_function(sample, x):
    counter_array = []
    n = len(sample)
    m = len(x)
    for i in range(m):
        counter_array.append(0)

    for i in range(m):
        for j in range(n):
            if sample[j] < x[i]:
                counter_array[i] = counter_array[i] + 1
        counter_array[i] = counter_array[i] / n
    return counter_array


def kernel_function(sample, h, x):
    res_array = []
    n = len(sample)
    m = len(x)

    for i in range(m):
        res_array.append(0)

    for i in range(m):
        for j in range(n):
            res_array[i] += normalized_distribution((x[i] - sample[j]) / h)
        res_array[i] = res_array[i] / (n * h)

    return res_array


def draw_empirical(sample, func, sector):
    if sector == 3:
        plt.title('Empirical distribution function for 20, 60, 100 elements.Distribution:' + func)
    plt.subplot(130+sector)
    if func == 'poisson':
        xx = np.linspace(0, 30, 100)
    else:
        xx = np.linspace(-4, 4, 100)
    plt.plot(xx, func_cumulative_dict[func](xx), 'b')
    plt.plot(xx, empirical_function(sample, xx), 'r')


def research_empirical(distribution_type):
    plt.figure("distribution " + distribution_type)
    num = 20
    sector = 1
    for i in range(3):
        sample = generate_dict[distribution_type](num)
        draw_empirical(sample, distribution_type, sector)
        num += 40
        sector += 1
    plt.show()


def draw_kernel(sample, func, sector, h):
    if sector == 3:
        plt.title('n = ' + str(len(sample)) + '. Kernel density estimation for h = [0.3, 0.6, 1.2]. Distribution: ' + func)
    plt.subplot(130+sector)
    if func == 'poisson':
        xx = np.linspace(0, 20, 100)
    else:
        xx = np.linspace(-4, 4, 100)
    plt.plot(xx, func_density_dict[func](xx), 'r')
    plt.plot(xx, kernel_function(sample, h,  xx), 'black')


def research_kernel(distribution_type):
    num = 20
    sector = 1
    h = 0.3
    for i in range(3):
        sample = generate_dict[distribution_type](num)
        plt.figure("distribution " + distribution_type + ", sample size: " + str(len(sample)))
        for j in range(3):
            draw_kernel(sample, distribution_type, sector, h)
            sector += 1
            h *= 2
        sector = 1
        num += 40
        h = 0.3
    plt.show()

#research_empirical('normal')
#research_empirical('laplace')
#research_empirical('uniform')
#research_empirical('cauchy')
#research_empirical('poisson')

#research_kernel('normal')
#research_kernel('laplace')
#research_kernel('uniform')
#research_kernel('cauchy')
research_kernel('poisson')

