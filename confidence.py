from scipy import stats
from numpy import *

def main():
    s = 3
    n = 10
    c1, c2 = stats.chi2.ppf([0.025, 1 - 0.025], n)
    y = zeros(50000)
    for i in range(len(y)):
        y[i] = sqrt(mean((random.randn(n) * s) ** 2))

    print("1-alpha=%.2f" % (mean((sqrt(n / c2) * y < s) & (sqrt(n / c1) * y > s)),))

if __name__ == "__main__":
    main()

