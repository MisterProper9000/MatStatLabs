import math
import sys

import numpy as np
import scipy.stats as ss

from collections import namedtuple


Range = namedtuple('Range', ['left', 'right'])

DISTRIBUTION_SIZE = 100
MIN_AMOUNT_IN_RANGE = 5


class DistributionFunction:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ss.norm.cdf(x, loc=self.mean, scale=self.std)


def generate_normal_distribution(size):
    return np.random.standard_normal(size)


def generate_range_frequency_probability_lists(distr, empirical_distr_func):
    def fill_range_list(min, max, cur_step):
        num = int((max - min) / cur_step)
        result = list()
        result.append(Range(-float('inf'), min + cur_step))
        for i in range(0, num - 2):
            result.append(Range(result[i].right, result[i].right + cur_step))
        result.append(Range(result[num - 2].right, float("inf")))

        return result

    def fill_probability_list(range_list, distr_func):
        result = list()
        for range in range_list:
            result.append(distr_func(range.right) - distr_func(range.left))
        return result

    def fill_frequency_list(range_list, distr_list):
        result = list()
        for range in range_list:
            freq = 0
            for val in distr_list:
                if range.left <= val <= range.right:
                    freq += 1
            result.append(freq)
        return result

    def check(size, prob_list):
        for i in range(0, len(prob_list)):
            if size * prob_list[i] < MIN_AMOUNT_IN_RANGE:
                return False
        return True

    size = len(distr)
    min_x = min(distr)
    max_x = max(distr)

    step = (max_x - min_x) / size

    cur_step = step
    should_stop = False
    while not should_stop:
        range_list = fill_range_list(min_x, max_x, cur_step)
        probability_list = fill_probability_list(range_list, empirical_distr_func)
        frequency_list = fill_frequency_list(range_list, distr)

        if not check(size, probability_list):
            cur_step += step
        else:
            should_stop = True

    return range_list, probability_list, frequency_list


def get_distribution_parameters(distr):
    mu = np.mean(distr)
    sum = 0
    for i in range(0, len(distr)):
        sum += math.pow(distr[i] - mu, 2)
    squared_sigma = sum / len(distr)

    return mu, math.sqrt(squared_sigma)


def calculate_pirson_criterion(distr_size, range_list, prob_list, freq_list):
    criterion = 0
    all_freq = 0
    all_prob = 0
    all_np = 0
    all_n_i_minus_np = 0
    for i in range(0, len(range_list)):
        np_i = distr_size * prob_list[i]
        n_i_minus_np_i = freq_list[i] - np_i
        coef = math.pow(n_i_minus_np_i, 2) / np_i

        criterion += coef
        all_freq += freq_list[i]
        all_prob += prob_list[i]
        all_np += np_i
        all_n_i_minus_np += n_i_minus_np_i

        print("%i; %.4f - %.4f; %i; %.4f; %.4f; %.4f; %.4f" %
              (i + 1,
               range_list[i].left, range_list[i].right,
               freq_list[i],
               prob_list[i],
               np_i,
               n_i_minus_np_i,
               coef
               ))

    print(";;%i;%.4f;%.4f;%.4f" % (all_freq, all_prob, all_np, all_n_i_minus_np))

    return criterion


if __name__ == '__main__':
    distr = generate_normal_distribution(DISTRIBUTION_SIZE)

    mu, sigma = get_distribution_parameters(distr)
    empirical_distr_func = DistributionFunction(mu, sigma)

    range_list, prob_list, freq_list = generate_range_frequency_probability_lists(distr, empirical_distr_func)
    k = len(range_list)

    f = open('out.csv', 'w')
    sys.stdout = f

    criterion = calculate_pirson_criterion(len(distr), range_list, prob_list, freq_list)

    print()
    print("%.4f;%.4f" % (k, criterion))
    print()
    print("%.4f;%.4f" % (mu, sigma))
