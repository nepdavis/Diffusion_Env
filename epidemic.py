import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit


class epidemic:

    def __init__(self, filename = None):

        if not filename:
            print('Error: must input filename.')

        try:

            if filename[-3:] == 'csv':
                delimiter = ','
            else:
                delimiter = '\t'

            self.data = np.genfromtxt(filename, delimiter=delimiter)

        except OSError:

            print('Error: file not found. Please input valid filename.')

        try:

            self.time = self.data[1:, 0]
            self.response = self.data[1:, 1]
            self.title = filename[:-4]

        except IndexError:

            print('Error: incorrect file format. Please provide either csv or txt.')

        self.filename = filename

        self.n = (100+(99*(len(self.time)-1)))
        self.i0 = self.response[0]
        self.s0 = self.n - self.i0
        self.r0 = 0

        self.fitted = None

        self.xpred = np.linspace(1, max(self.time), 1000)

        self.lst1 = np.array([self.response[0]] + [self.response[i] - self.response[i - 1] for i in range(1, len(self.response))])


    def __sir__(self, y, t, b, g):

        s = -b * y[0] * y[1] / self.n
        r = g * y[1]
        i = -(s + r)

        return s, i, r

    def integrate(self, t, b, g):

        return odeint(self.__sir__, (self.s0, self.i0, self.r0), t, args=(b, g))[:,1]

    def fit(self):

        popt, pcov = curve_fit(self.integrate, self.time, self.lst1)

        self.fitted = self.integrate(self.xpred, *popt)

        plt.plot(self.time, self.lst1, 'o')
        plt.plot(self.xpred, self.fitted)
        plt.show()




model = epidemic('ALS.csv')
model.fit()
