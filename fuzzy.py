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

from __future__ import print_function
import numpy as np

#
# python version of the fuzzy rbm supports the non-fuzzy version.
#
# use pickle.dump(instance,file) and pickle.load(file) to save and restore data.
# file is a Python FILE object so it's opened first.
#
# fuzzy uses an ndarray so the ndarray primitives are available
# we'll wrap some of them into a standard interface, but you can
# roll your own if need be.
#


class Fuzzy(object):
    ''' Python version of the fuzzy RBM supports the
    non-fuzzy version. '''

    def __init__(self, the_min, the_max, num_dimensions):
        self.my_min = the_min
        self.my_max = the_max
        self.delta = (the_max-the_min)/num_dimensions
        self.num_dimensions = num_dimensions
        self.counts = np.float32(np.zeros(num_dimensions))
        ddi = num_dimensions/2
        self.args = [(i-ddi)*self.delta for i in range(num_dimensions)]

    def initialize_counts(self):
        self.counts.__imul__(0.)

    def add(self, what):
        i = int((what - self.my_min)/self.delta + 0.5)
        # insert rangechecking here.
        if i >= self.num_dimensions:
            i = self.num_dimensions - 1
        if i < 0:
            i = 0
        self.counts[i] += 1.

    def expected_value(self):
        ds = self.counts.sum()
        if ds == 0.:
            ds = 1.
        dsum = np.dot(self.args, self.counts)
        return dsum/ds

    def belief(self):
        ds = self.counts.sum()
        if ds == 0.:
            return (self.my_min + self.my_max)*0.5, 0.
        # use numpy you dumb fsck
        im = float(np.argmax(self.counts))
        return (im*self.delta + self.my_min), im/ds

    def damp(self, avalue):
        ds = self.counts.sum()
        if ds == 0.:
            return 1.
        im = int((avalue - self.my_min)/(self.my_max-self.my_min)+0.5)
        if im >= self.num_dimensions:
            im = self.num_dimensions - 1
        if im < 0:
            im = 0
        i = np.argmax(self.counts)
        if abs(i-im) < 2:
            return 1.
        return -0.1


def main():
    print("This is the main routine, defined for testing purposes")
    simon = Fuzzy(-1., 1., 10)
    simon.add(0.)
    simon.add(0.1)
    simon.add(0.2)
    simon.add(0.3)
    print(simon.counts)
    print(simon.expected_value())
    simon.initialize_counts()
    print(simon.counts)
    simon.add(0.)
    simon.add(0.1)
    simon.add(0.2)
    print(simon.counts)
    print(simon.expected_value())


if __name__ == '__main__':
    main()
