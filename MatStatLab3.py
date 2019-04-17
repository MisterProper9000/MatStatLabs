
import numpy
import plotly.graph_objs as go
import plotly.plotly as py
import seaborn as sns
import matplotlib.pyplot as mplot
import sys


POISSON_PARAM = 2
UNIFORM_LEFT = -numpy.sqrt(3)
UNIFORM_RIGHT = numpy.sqrt(3)
LAPLAS_COEF = numpy.sqrt(2)
selection = [20, 100]
selection = numpy.sort(selection)

def standart_normal(x):
    return (1 / numpy.sqrt(2*numpy.pi)) * numpy.exp(- x * x / 2)


def standart_cauchy(x):
    return 1 / (numpy.pi * (1 + x*x))


def laplace(x):
    return 1 / LAPLAS_COEF * numpy.exp (-LAPLAS_COEF * numpy.abs(x))


def uniform(x):
    flag2 = x <= UNIFORM_RIGHT
    flag1 = x >= UNIFORM_LEFT
    return 1 / (UNIFORM_RIGHT - UNIFORM_LEFT) * flag1 * flag2


def poisson(x):
    k = POISSON_PARAM
    return (numpy.power(x, k) / numpy.math.factorial(k)) * numpy.exp(-x)


func_dict = {
    'normal': standart_normal,
    'cauchy': standart_cauchy,
    'laplace': laplace,
    'uniform': uniform,
    'poisson': poisson
}


def generate_laplace(x):
    return numpy.random.laplace(0, 1/LAPLAS_COEF, x)


def generate_uniform(x):
    return numpy.random.uniform(UNIFORM_LEFT, UNIFORM_RIGHT, x)


def generate_poisson(x):
    return numpy.random.poisson(POISSON_PARAM, x)


generate_dict = {
    'normal': numpy.random.standard_normal,
    'cauchy': numpy.random.standard_cauchy,
    'laplace': generate_laplace,
    'uniform': generate_uniform,
    'poisson': generate_poisson
}


def Zr(x):
    return (numpy.amin(x) + numpy.amax(x))/2


def Zq(x):
    return (numpy.quantile(x, 1/4) + numpy.quantile(x, 3/4)) / 2


def Ztr(x):
    length = x.size
    r = (int)(length / 4)
    sum = 0
    for i in range(r, length - r):
        sum += x[i]
    return sum/(length - 2 * r)


def IQR(x):
    return numpy.abs(numpy.quantile(x, 1 / 4) - numpy.quantile(x, 3 / 4))


def ejection(x):
    length = x.size
    count = 0
    left = numpy.quantile(x, 1 / 4) - 1.5 * IQR(x)
    right = numpy.quantile(x, 3 / 4) + 1.5 * IQR(x)
    for i in range(0, length):
        if(x[i] < left or x[i] > right):
            count += 1
    return count / length


pos_characteristic_dict = {
    'average': numpy.mean,
    'med': numpy.median,
    'Zr': Zr,
    'Zq': Zq,
    'Ztr r = n/4': Ztr
}

pos_char_name = [
    'average',
    'med',
    'Zr',
    'Zq',
    'Ztr r = n/4'
]


def E(z):
    return numpy.mean(z)


def D(z):
    return numpy.var(z)


f = open('out1.csv', 'w')
std = sys.stdout
sys.stdout = f

def research(dist_type):
   # print('-------------------------------------')
    print()
    print(dist_type)

    data = []

    for num in selection:
        eject = []
        arr = numpy.sort(generate_dict[dist_type](num))
        data.append(arr)

        for i in range(0, 1000):
            arr = numpy.sort(generate_dict[dist_type](num))
            eject.append(ejection(arr))

        print("%-10s;" % ('n = %i' % num), end="")
        print("%-12f;" % E(eject), end="")
        print()

    mplot.figure(dist_type)
    mplot.title(dist_type)
    sns.set(style="whitegrid")
    ax = sns.boxplot(data=data, orient='h')
    mplot.yticks(numpy.arange(2), ('20', '100'))
    mplot.show()




research('normal')
research('cauchy')
research('laplace')
research('uniform')
research('poisson')


f.close()
sys.stdout = std
print("Done")