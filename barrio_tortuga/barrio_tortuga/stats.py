import numpy as np
from   typing  import Tuple, List
from numpy import sqrt

NN = np.nan

from scipy.stats import nbinom
from scipy.stats import weibull_min
from scipy.stats import norm
from scipy.stats import skewnorm
from scipy.stats import lognorm


def c19_nbinom_transform(r0, k):
    """Transforms the definition used in C19 analysis to standard"""
    n = k
    p = (1 + r0/k)**(-1)
    return n, p


def c19_nbinom_pdf(r0, k, x):
    """PDF of the negative binomial used in c19 studies"""
    n, p = c19_nbinom_transform(r0, k)
    return nbinom.pmf(x, n, p)


def c19_nbinom_rvs(r0, k, size=10):
    """Generates random variates"""
    n, p = c19_nbinom_transform(r0, k)

    if size > 1:
        r= nbinom.rvs(n, p, size=size)
    else:
        r= nbinom.rvs(n, p)
    return r


def weib_pdf(x,n,a):
    """Weibull distribution PDF with shape a and scale n"""
    return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)


def c19_weib_rvs(mu, rms, size=10):
    """Generates random variates"""
    r = mu * np.random.weibull(rms, size=size)
    return r


def normal_pdf(x,mu,rms):
    return norm.pdf(x, loc=mu, scale=rms)


def normal_rvs(mu, sigma, size=10):
    """Generates random variates"""
    return norm.rvs(loc=mu, scale=sigma, size=size)


def sknormal_pdf(x,mu,rms, a):
    return skewnorm.pdf(x, a, loc=mu, scale=rms)


def sknormal_rvs(mu, sigma, a, size=0):
    """Generates random variates"""
    if size == 0:
        return skewnorm.rvs(a, loc=mu, scale=sigma)
    else:
        return skewnorm.rvs(a, loc=mu, scale=sigma, size=size)


def lognorm_pdf(x, mu, sigma):
    """lognorm distribution"""
    return lognorm.pdf(x, sigma, scale=np.exp(mu))
