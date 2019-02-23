import numpy as np
import matplotlib.pyplot as plt
Lambda = 7
Bound = np.sqrt(3)
def normalized_distr(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-x * x / 2)

def laplace_distr(x):
    return (1 / np.sqrt(2)) * np.exp(-np.sqrt(2) * np.abs(x))

def cauchy_distr(x):
    return 1 / (np.pi * (1 + x * x))

def poisson_distr(x):
    return (np.power(x, Lambda) / np.math.factorial(Lambda)) * np.exp(-x)

def uniform_distr(x):
    return 1 / (2 * Bound) * (x <= Bound)

def laplace_gen(x):
    return np.random.laplace(0, 1/np.sqrt(3), x)

def poisson_gen(x):
    return np.random.poisson(Lambda, x)

def uniform_gen(x):
    return np.random.uniform(-Bound, Bound, x)

distrs = {
    'normal'  : normalized_distr,
    'laplace' : laplace_distr,
    'cauchy'  : cauchy_distr,
    'poisson' : poisson_distr,
    'uniform' : uniform_distr,
}

generate_dict = {
    'normal'  : np.random.standard_normal,
    'laplace' : laplace_gen,
    'cauchy'  : np.random.standard_cauchy,
    'poisson' : poisson_gen,
    'uniform' : uniform_gen,
}

def draw(array, func, chunk, i):
    plt.subplot(221 + chunk)
    plt.tight_layout()
    plt.hist(array, 15, density=True)
    xx = np.linspace(np.min(array), np.max(array), 100)
    plt.plot(xx, distrs[func](xx), 'r')
    plt.title('n = %i' %i)

distrs_name = ['normal', 'laplace', 'cauchy', 'poisson', 'uniform']
num = [10,50,100,1000]
for name in range(5):
    plt.figure("distribution " + distrs_name[name])
    plt.title('distribution %s' %distrs_name[name])
    chunk = 0
    for i in range(4):
        draw(generate_dict[distrs_name[name]](num[i]), distrs_name[name], chunk, num[i])
        chunk += 1
    plt.show()