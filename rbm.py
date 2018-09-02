''' 
Copyright (c) 2017  Treadco LLC, Amelia Treader, Robert W Harrison

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import sys
import os
from math import exp as exp

import fuzzy

# python version of the fuzzy rbm
# supports the non-fuzzy version.
#
# use pickle.dump(instance,file)
#  and pickle.load(file)
#
# to save and restore data. file is a Python FILE object so
# it's opened first.

class RBM:  # the basic rbm
    def __init__(self, number_visible, number_hidden):
        self.fuzz = []
        self.nvis = number_visible
        self.nhid = number_hidden
        self.layers = []
        self.energies = []
        self.hidden = []
        self.symmetric_encoding = False
        # initialize the space
        # making essentially empty lists means that we can avoid using append etc
        self.scratch = np.full(number_visible, 0.0)
        for _ in range(self.nhid):
            self.layers.append(np.float32(np.zeros(self.nvis)))
            self.hidden.append(0)
            self.energies.append(0.)

    def add_fuzzy(self, thmin, thmax, thenumber):
        self.fuzz = [fuzzy.Fuzzy(thmin, thmax, thenumber)
                     for x in range(self.nhid)]

    def reinitialize_fuzzy(self):
        self.fuzz = [x.initialize_counts() for x in self.fuzz]

    def reconstruct(self, data, use_best=True):
        the_layer = self.the_best_layer(data, use_best)
        ib = the_layer[0]
        a = self.layers[ib]
        sign = 1.
        if self.hidden[ib] < 0.:
            sign = -1.
        #
        # there may be a clever numpy solution for this loop
        #
        for i in range(0, self.nvis):
            self.scratch[i] = -1.
            if a[i] < 0.:
                self.scratch[i] = 1.
        return self.scratch.__mul__(sign)

    def its_symmetric(self):
        self.symmetric_encoding = True

    def the_best_layer(self, data, use_best=True):
        if use_best:
            self.assign_hidden_and_reconstruction_energy(data)
        else:
            self.assign_hidden_and_energy(data)
        ib = np.argmin(self.energies)
        eb = self.energies[ib]
        return ib, eb

    def the_best_built_layer(self, data, use_best=True):
        if use_best:
            self.assign_hidden_and_reconstruction_energy(data)
        else:
            self.assign_hidden_and_energy(data)
        ib = np.argmin(self.energies)
        eb = self.energies[ib]
        while use_best and eb < -1.:
            self.energies[ib] = 10.e10
            ib = np.argmin(self.energies)
            eb = self.energies[ib]
        return ib, eb

    def estimate_expected_value(self, data, use_best=True):
        ib = self.the_best_layer(data, use_best)[0]
        return self.fuzz[ib].expected_value()

    def assign_hidden_and_reconstruction_energy(self, data):
        for i in range(0, self.nhid):
            eraw = np.dot(data, self.layers[i])
            ebest = np.dot(data.__abs__(), (self.layers[i]).__abs__())
            if ebest == 0.0:
                # this forces the RBM to train this layer.
                self.energies[i] = -10.e10
                self.hidden[i] = 1.0
            elif self.symmetric_encoding:
                self.hidden[i] = 1.0
                self.energies[i] = eraw/ebest
            else:
                if eraw > 0.:
                    self.hidden[i] = -1.0
                    self.energies[i] = -eraw/ebest
                else:
                    self.hidden[i] = 1.0
                    self.energies[i] = eraw/ebest

    def assign_hidden_and_energy(self, data):
        for i in range(0, self.nhid):
            eraw = np.dot(data, self.layers[i])
            if self.symmetric_encoding:
                self.hidden[i] = 1.0
                self.energies[i] = eraw
            else:
                if eraw > 0.:
                    self.hidden[i] = -1.0
                    self.energies[i] = -eraw
                else:
                    self.hidden[i] = 1.0
                    self.energies[i] = eraw

    def trainOmatic(self, data, beta, learning_rate, use_best=True):
        self.train(data, beta, learning_rate, use_best)
        self.antitrain(data, beta, learning_rate*0.1, use_best)

    def train_fuzzy(self, data, beta, learning_rate, dependent_value, use_best=True):
        """ this is the online one pass algorithm. """
        if len(self.fuzz) == 0:
            print("You Must define fuzzy first to use this")
        if use_best:
            self.assign_hidden_and_reconstruction_energy(data)
        else:
            self.assign_hidden_and_energy(data)
        # select the row to train.
        imin = 0
        emin = self.energies[0]
        for i in range(1, self.nhid):
            if emin >= self.energies[i]:
                imin = i
                emin = self.energies[i]
        #
        # emin,imin now point to the best row
        #
        hsign = self.hidden[imin]
        alayer = self.layers[imin]
        self.fuzz[imin].add(dependent_value)
        # the products with hsign keep the +- straight.
        # for the gradients that is.
        # learning_rate = learning_rate*fdamp
        for i in range(0, self.nvis):  # over the row
            ef = alayer[i]*hsign*data[i]
            ep = ef*beta*hsign
            em = -ep
            fp = exp(-ep)
            fm = exp(-em)
            damp = (fp-fm)/(fp+fm) * hsign * data[i]
            hv = hsign * data[i]
            alayer[i] += learning_rate*(-hv + damp)
        return emin

    def train(self, data, beta, learning_rate, use_best=True):
        if use_best:
            self.assign_hidden_and_reconstruction_energy(data)
        else:
            self.assign_hidden_and_energy(data)
        # select the row to train.
        imin = 0
        emin = self.energies[0]
        for i in range(1, self.nhid):
            if emin >= self.energies[i]:
                imin = i
                emin = self.energies[i]
        # emin,imin now point to the best row
        hsign = self.hidden[imin]
        alayer = self.layers[imin]
        # the products with hsign keep the +- straight.
        # for the gradients that is.
        for i in range(0, self.nvis):  # over the row
            ef = alayer[i]*hsign*data[i]
            ep = ef*beta*hsign
            em = -ep
            fp = exp(-ep)
            fm = exp(-em)
            damp = (fp-fm)/(fp+fm) * hsign * data[i]
            hv = hsign * data[i]
            alayer[i] += learning_rate*(-hv + damp)
        return emin

    def antitrain(self, data, beta, learning_rate, use_best=True):
        if use_best:
            self.assign_hidden_and_reconstruction_energy(data)
        else:
            self.assign_hidden_and_energy(data)
        # select the row to train.
        imax = 0
        emax = self.energies[0]
        for i in range(1, self.nhid):
            if emax <= self.energies[i]:
                imax = i
                emax = self.energies[i]
        #
        # emin,imin now point to the best row
        #
        hsign = self.hidden[imax]
        alayer = self.layers[imax]
        # the products with hsign keep the +- straight.
        # for the gradients that is.
        for i in range(0, self.nvis):  # over the row
            ef = alayer[i]*hsign*data[i]
            ep = ef*beta*hsign
            em = -ep
            fp = exp(-ep)
            fm = exp(-em)
            damp = (fp-fm)/(fp+fm) * hsign * data[i]
            hv = hsign * data[i]
            alayer[i] -= learning_rate*(-hv + damp)


def main():
    print("this is the main routine, set up for testing")
    my_rbm = RBM(2, 2)
    print(my_rbm.layers)
    d = np.full(2, 1.)
    d[0] = -1.0
    #   my_rbm.train(d, 0.1, 0.1)
    #   print(my_rbm.layers)
    for _ in range(1, 10):
        d[0] = 1.
        d[1] = -1.
        my_rbm.train(d, 0.1, 0.1)
        d[0] = 1.
        d[1] = 1.
        my_rbm.train(d, 0.1, 0.1)
        print(my_rbm.layers)

    d[0] = 0.
    print(my_rbm.reconstruct(d))
    d[0] = 1.
    d[1] = 0.
    print(my_rbm.reconstruct(d))

    my_rbm.layers[0] = np.array([-1., 1.])
    my_rbm.layers[1] = np.array([1., 1.])
    my_rbm.add_fuzzy(-1., 1., 20)
    print(my_rbm.layers)
    d[0] = 1.
    d[1] = -1.
    my_rbm.train_fuzzy(d, 0.1, 0.1, 0.4)
    d[0] = 1.
    d[1] = 1.
    my_rbm.train_fuzzy(d, 0.1, 0.1, -0.4)
    print(my_rbm.layers)
    print(d, my_rbm.estimate_expected_value(d))
    d[0] = 1.
    d[1] = -1.
    print(d, my_rbm.estimate_expected_value(d))


if __name__ == '__main__':
    main()
