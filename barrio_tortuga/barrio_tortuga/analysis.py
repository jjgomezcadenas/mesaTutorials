import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def peak_position(dft, ticks_per_day=1):
    return dft.NumberOfInfected.idxmax()/ticks_per_day, dft.NumberOfInfected.max()


def r0(dft, tmax=25, tr=5, ticks_per_day=5):
    T = dft.index.values
    E = dft.NumberOfExposed.values
    I = dft.NumberOfInfected.values
    S = dft.NumberOfSusceptible.values
    R = dft.NumberOfRecovered.values
    N = S[0] + I[0]
    r0 = np.array([(E[t] - E[t-1]) / (S[t-1]/N) / I[t-1] for t in T[1:tmax]])
    return T[1:tmax], r0 * tr * ticks_per_day


def r0_series(DFD, tmax=25, tr=5, ticks_per_day=5):
    R0D = {}
    AR0D = {}
    TD  = {}

    for key, value in DFD.items():
        TD[key], R0D[key] = r0(value, tmax, tr, ticks_per_day)
        AR0D[key] = R0D[key].mean()
    return TD, R0D, AR0D




def plot_average_I(DFD, F=True, S=True, T=' Infected: R0 = 3.5, ti = 5.5, tr = 5', figsize=(8,8)):
    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(111)

    ftot = DFD['F:average'].NumberOfInfected[0] + DFD['F:average'].NumberOfSusceptible[0]
    stot = DFD['S:average'].NumberOfInfected[0] + DFD['S:average'].NumberOfSusceptible[0]
    if F:
        plt.plot(DFD['F:average'].index, DFD['F:average'].NumberOfInfected/ftot, 'r', lw=2, label='Fixed' )
    if S:
        plt.plot(DFD['S:average'].index, DFD['S:average'].NumberOfInfected/stot, 'b', lw=2, label='Stochastic'  )
    plt.xlabel('time (days)')
    plt.ylabel('Fraction of Infected')
    plt.legend()
    plt.title(T)
    plt.show()


def plot_runs_I(DFD, F=True, S=True, T='Infected: R0 = 3.5, ti = 5.5, tr = 5', figsize=(8,8)):
    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(111)

    ftot = DFD['F:average'].NumberOfInfected[0] + DFD['F:average'].NumberOfSusceptible[0]
    stot = DFD['S:average'].NumberOfInfected[0] + DFD['S:average'].NumberOfSusceptible[0]

    for key, value in DFD.items():
        name = key.split(":")
        if name[0] == 'F' and name[1] != 'average' and F:
            plt.plot(value.index, value.NumberOfInfected/ftot, lw=2, label=key  )
        if name[0] == 'S' and name[1] != 'average' and S:
            plt.plot(value.index, value.NumberOfInfected/stot, lw=2, label=key  )

    plt.xlabel('time (days)')
    plt.ylabel('Fraction of Infected')
    plt.legend()
    plt.title(T)
    plt.show()


def plot_I_E(DFD, F=True, S=True, T=' Infected/Exposed: R0 = 3.5, ti = 5.5, tr = 5',
             figsize=(8,8)):
    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(111)

    ftot = DFD['F:average'].NumberOfInfected[0] + DFD['F:average'].NumberOfSusceptible[0]
    stot = DFD['S:average'].NumberOfInfected[0] + DFD['S:average'].NumberOfSusceptible[0]
    if F:
        plt.plot(DFD['F:average'].index, DFD['F:average'].NumberOfInfected/ftot, 'r', lw=2,
        label='I : Fixed' )
        plt.plot(DFD['F:average'].index, DFD['F:average'].NumberOfExposed/ftot, 'b', lw=2,
        label='E : Fixed' )
    if S:
        plt.plot(DFD['S:average'].index, DFD['S:average'].NumberOfInfected/stot, 'g', lw=2,
        label='I : Stochastic'  )
        plt.plot(DFD['S:average'].index, DFD['S:average'].NumberOfExposed/stot, 'y',  lw=2,
        label='E : Stochastic'  )
    plt.xlabel('time (days)')
    plt.ylabel('Fraction of Infected')
    plt.legend()
    plt.title(T)
    plt.show()


def plot_S_R(DFD, F=True, S=True, T=' R0 = 3.5, ti = 5.5, tr = 5', figsize=(8,8)):
    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(111)

    ftot = DFD['F:average'].NumberOfInfected[0] + DFD['F:average'].NumberOfSusceptible[0]
    stot = DFD['S:average'].NumberOfInfected[0] + DFD['S:average'].NumberOfSusceptible[0]
    if F:
        plt.plot(DFD['F:average'].index, DFD['F:average'].NumberOfSusceptible/ftot, 'r', lw=2, label='S : Fixed' )
        plt.plot(DFD['F:average'].index, DFD['F:average'].NumberOfRecovered/ftot, 'b', lw=2, label  ='R : Fixed' )
    if S:
        plt.plot(DFD['S:average'].index, DFD['S:average'].NumberOfSusceptible/stot, 'g', lw=2, label='S : Stochastic'  )
        plt.plot(DFD['S:average'].index, DFD['S:average'].NumberOfRecovered/stot, 'y',  lw=2, label ='R : Stochastic'  )
    plt.xlabel('time (days)')
    plt.ylabel('Fraction of Infected')
    plt.legend()
    plt.title(T)
    plt.show()
