import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

class Interval:
    def __init__(self):
        Interval.a = (int)
        Interval.b = (int)


def assimptot_method(m, d, n, ar):
    t = 1.96
    shift_ar_2 = [(ar[i] - m) ** 2 for i in range(0, n)]
    shift_ar_4 = [(ar[i] - m) ** 4 for i in range(0, n)]
    M4 = sum(shift_ar_4) / n
    M2 = sum(shift_ar_2) / n
    e = ((n**2 - 1) / ((n - 2) * (n - 3))) * ((M4 / (M2)**2) - 3 + (6/(n + 1)) )
    interval_m = Interval()
    interval_m.a = m - (t * d / (n)**0.5)
    interval_m.b = m + (t * d / (n)**0.5)
    interval_d = Interval()
    interval_d.a = d * (1 - 0.5 * t * ((e + 2) / n)**0.5)
    interval_d.b = d * (1 + 0.5 * t * ((e + 2) / n)**0.5)
    print("Нормальный интервал n = " + str(n))
    print("Мат.ожидание")
    print(interval_m.a, interval_m.b)
    print("Дисперсия")
    print(interval_d.a, interval_d.b)


def classic_method(m, d, n):
    if n == 20:
        a1 = 8.91
        a2 = 32.9
        t = 1.7291
    else:
        a1 = 73.46108
        a2 = 128.422
        t = 1.9842
    interval_m = Interval()
    interval_m.a = m - (t * d / (n - 1)**0.5)
    interval_m.b = m + (t * d / (n - 1) ** 0.5)
    interval_d = Interval()
    interval_d.a = d * (n - 1)**0.5 / (a2)**0.5
    interval_d.b = d * (n - 1)**0.5 / (a1)**0.5
    print("Классический интервал n = " + str(n))
    print("Мат.ожидание - Сьютенд")
    print(interval_m.a, interval_m.b)
    print("Дисперсия - Хи-квадрат")
    print(interval_d.a, interval_d.b)


def main():
    ar_20 = np.random.randn(20)
    ar_100 = np.random.randn(100)
    m_20, m_100 = stat.mean(ar_20), stat.mean(ar_100)
    shift_ar_20 = [(ar_20[i] - m_20) ** 2 for i in range(0, len(ar_20))]
    shift_ar_100 = [(ar_100[i] - m_100) ** 2 for i in range(0, len(ar_100))]
    d_20 = (sum(shift_ar_20) / (len(shift_ar_20) - 1)) ** 0.5
    d_100 = (sum(shift_ar_100) / (len(shift_ar_100) - 1)) ** 0.5
    classic_method(m_20, d_20, 20)
    classic_method(m_100, d_100, 100)
    assimptot_method(m_20, d_20, 20, ar_20)
    assimptot_method(m_100, d_100, 100, ar_100)


main()



