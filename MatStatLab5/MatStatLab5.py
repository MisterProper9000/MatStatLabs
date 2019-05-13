import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.patches import Ellipse
import sys

linalg = np.linalg

selection = [20, 60, 100, 1000]
cor = [0, 0.5, 0.9]






def E(x):
    return np.mean(x)


def D(x):
    return np.var(x)


def sqE(x):
    length = len(x)
    sum = 0
    for x_i in x:
        sum += x_i ** 2
    return sum / length


def pearson(x, y):
    res = stats.pearsonr(x, y)
    return res[0]


def spearman(x, y):
    res = stats.spearmanr(x, y)
    return res[0]


def quad(x, y):
    length = len(x)
    med_x = np.median(x)
    med_y = np.median(y)
    sum = 0
    for i in range(0, length):
        sum = sum + np.sign(x[i] - med_x) * np.sign(y[i] - med_y)
    return sum / length


cor_coef_dict = {
    "Pearson": pearson,
    "Spearman": spearman,
    "Quad": quad
}



def mix_normal_dist(p, N):
    cov1 = [[1, p], [p, 1]]
    cov2 = [[100, -p], [-p, 100]]
    mean = [0, 0]
    return 0.9 * np.random.multivariate_normal(mean, cov1, N) + 0.1 * np.random.multivariate_normal(mean, cov2, N)


def normal_dist(p, N):
    cov = [[1, p], [p, 1]]
    mean = [0, 0]
    return np.random.multivariate_normal(mean, cov, N)



dist_dict = {
    "Normal": normal_dist,
    "NormalMix": mix_normal_dist
}


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def dist_ellips(x, y, ax):
    nstd = 2.5
    #ax = plt.subplot(111)

    cov = np.cov(x, y)
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                  width=w, height=h,
                  angle=theta, color='black')
    ell.set_facecolor('none')
    ax.add_artist(ell)
    plt.scatter(x, y)



def research(p, N, dist, ax):
    pears = []
    spear = []
    quads = []
    for i in range(0, 10):
        data = dist(p, N)

        pears.append(pearson(data[:, 0], data[:, 1]))
        spear.append(spearman(data[:, 0], data[:, 1]))
        quads.append(quad(data[:, 0], data[:, 1]))

    plt.scatter(data[:, 0], data[:, 1], c='green')
    dist_ellips(data[:, 0], data[:, 1], ax)
    print("name;", end="")
    for name in cor_coef_dict:
        print("%-12s;" % name, end="")
    print()
    print("\t\tE   ;%.5lf;%.5lf;%.5lf" % (E(pears), E(spear), E(quads)))
    print("\t\tE^2 ;%.5lf;%.5lf;%.5lf" % (sqE(pears), sqE(spear), sqE(quads)))
    print("\t\tD   ;%.5lf;%.5lf;%.5lf" % (D(pears), D(spear), D(quads)))
    #print("\t\tE   : pears = %.5lf, spear = %.5lf, quad = %.5lf" % (E(pears), E(spear), E(quads)))
    #print("\t\tE^2 : pears = %.5lf, spear = %.5lf, quad = %.5lf" % (sqE(pears), sqE(spear), sqE(quads)))
    #print("\t\tD   : pears = %.5lf, spear = %.5lf, quad = %.5lf" % (D(pears), D(spear), D(quads)))
    print()



def draw(dist_name, p):
    plt.figure(p)
    sector = 1
    for N in selection:
        ax = plt.subplot(220 + sector)

        sector += 1
        if(dist_name == "NormalMix"):
            plt.title(dist_name + " n=%i" % N + " p1=%.1f" % p + " p2=%.1f" % (-p))
            print(dist_name + ", n=%i" % N + ", p1=%.1f" % p + ", p2=%.1f" % (-p))
        else:
            plt.title(dist_name + ", n=%i" % N + ", p=%.1f" % p)
            print(dist_name + ", n=%i" % N + ", p=%.1f" % p)

        research(p, N, dist_dict[dist_name], ax)

    plt.show()





f = open('out.csv', 'w')
sys.stdout = f

for p in cor:
    draw("Normal", p)
    print()

draw("NormalMix", 0.9)
