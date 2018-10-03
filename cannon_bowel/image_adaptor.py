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
from random import random
import numpy as np

class Adaptor(object):
    def __init__(self, half_width, nsigma, an_array):
        self.width = 1 + 2*half_width  # must be odd x odd to work
        # keep this so I can undo it later if I want
        self.mean = an_array.mean()
        self.scale = an_array.std()
        self.nsigma = nsigma
        self.the_image = np.clip((an_array.__sub__(self.mean)).__div__(
            float(nsigma)*self.scale), -1., 1.)
        self.scratch = np.float32(np.zeros((self.width * self.width)))
        self.half = half_width
        self.ix = 0
        self.iy = 0
        self.nx = self.the_image.shape[0]
        self.ny = self.the_image.shape[1]
        self.from_bits = []
        self.range_delta = None
        self.nbits = None
        self.bits = None
        self.indexes = None

    def to_bits(self, depth):
        i = pow(2, depth)
        self.bits = []
        self.from_bits = [pow(2, j)*0.5 for j in range(depth)]
        k = -1.
        for j in range(0, i):
            self.bits.append([k])
            k = -k
        # did bit zero by itself because %1
        for l in range(1, depth):
            k = -1.
            sign = 0
            sign_mod = pow(2, l)
            for j in range(0, i):
                self.bits[j].append(k)
                sign = sign + 1
                if sign % sign_mod == 0:
                    k = -k
        self.range_delta = float(i-1)  # avoid picket fence
        self.nbits = depth

    def to_wavelet(self, depth):
        """ Hard wired for 8. If it works we'll put code for other 2^n
        8 orthogonal vectors. """
        self.bits = []
        self.bits.append([1., 1., 1., 1., 1., 1., 1., 1.])
        self.bits.append([1., 1., 1., 1., -1., -1., -1., -1.])
        self.bits.append([-1., -1., 1., 1., -1., -1., 1., 1.])
        self.bits.append([-1., -1., 1., 1., 1., 1., -1., -1.])
        self.bits.append([-1., 1., -1., 1., -1., 1., -1., 1.])
        self.bits.append([-1., 1., -1., 1., 1., -1., 1., -1.])
        self.bits.append([-1., 1., 1., -1., -1., 1., 1., -1.])
        self.bits.append([-1., 1., 1., -1., 1., -1., -1., 1.])
        self.range_delta = float(depth-1)  # avoid picket fence
        self.nbits = depth

    def bits_to_density(self, bits):
        ac = 0.
        for i in range(self.nbits):
            ac = ac + self.from_bits[i]*(bits[i] + 1.)
        return ac

    def make_nbit_image(self, depth):
        self.to_bits(depth)
        self.scratch = np.float32(np.zeros((self.width * self.width*depth)))
        ma = self.the_image.max()
        mi = self.the_image.min()
        self.indexes = np.uint8(np.zeros_like(self.the_image))
        for i in range(0, self.nx):
            for j in range(0, self.ny):
                self.indexes[i][j] = int(
                    (self.the_image[i][j]-mi)/(ma-mi)*self.range_delta)

    def to_bitmap(self, depth):
        i = depth
        self.bits = []
        self.from_bits = []
        dl = 1./float(depth)
        for j in range(0, depth):
            self.from_bits.append(j*dl)
            self.bits.append([])

        for l in range(0, depth):
            for j in range(0, depth):
                if j == l:
                    self.bits[l].append(1.)
                else:
                    self.bits[l].append(-1.)
            print(self.bits[l])
        self.range_delta = float(i-1)  # avoid picket fence
        self.nbits = depth

    def make_nbitmap_image(self, depth):
        self.to_bitmap(depth)
        self.scratch = np.float32(np.zeros((self.width*self.width*depth)))
        ma = self.the_image.max()
        mi = self.the_image.min()
        self.indexes = np.uint8(np.zeros_like(self.the_image))
        for i in range(0, self.nx):
            for j in range(0, self.ny):
                self.indexes[i][j] = int(
                    (self.the_image[i][j]-mi)/(ma-mi)*self.range_delta)

    def make_nwavelet_image(self, depth):
        self.to_wavelet(depth)
        self.scratch = np.float32(np.zeros((self.width*self.width*depth)))
        ma = self.the_image.max()
        mi = self.the_image.min()
        self.indexes = np.uint8(np.zeros_like(self.the_image))
        for i in range(0, self.nx):
            for j in range(0, self.ny):
                self.indexes[i][j] = int(
                    (self.the_image[i][j]-mi)/(ma-mi)*self.range_delta)

    def to_original(self, apixel):
        return apixel*self.scale*(float(self.nsigma)) + self.mean

    def reset(self):
        self.ix = 0
        self.iy = 0

    def random_bits(self):
        rx = int(random()*(self.nx - self.width))
        ry = int(random()*(self.ny - self.width))
        self.scratch = []
        for i in range(0, self.width):
            for j in range(0, self.width):
                self.scratch.append(self.bits[self.indexes[rx+i][ry+j]])
        self.scratch = np.array(sum(self.scratch, []))
        return (True, self.the_image[rx+self.half][ry+self.half], self.scratch)

    def random(self):
        rx = int(random()*(self.nx-self.width))
        ry = int(random()*(self.ny-self.width))
        for i in range(0, self.width):
            inx = i*self.width
            for j in range(0, self.width):
                self.scratch[inx + j] = self.the_image[rx+i][ry+j]
        return (True, self.the_image[rx+self.half][ry+self.half], self.scratch)

    def at_bits(self, rx, ry):
        self.scratch = []
        for i in range(0, self.width):
            for j in range(0, self.width):
                self.scratch.append(self.bits[self.indexes[rx+i][ry+j]])
        self.scratch = np.array(sum(self.scratch, []))
        return (True, self.the_image[rx+self.half][ry+self.half], self.scratch)

    def at(self, rx, ry):
        for i in range(0, self.width):
            inx = i*self.width
            for j in range(0, self.width):
                self.scratch[inx+j] = self.the_image[rx+i][ry+j]
        return (True, self.the_image[rx+self.half][ry+self.half], self.scratch)

    def next(self, dx, dy=0):
        jx = self.ix+dx
        jy = self.iy
        if dy == 0:
            if jx > self.nx - self.width:
                jy = self.iy + 1
                jx = 0
        else:
            jy = self.iy + dy
        if jy >= self.ny - self.width or jx >= self.nx - self.width:
            return (False, 0., self.scratch)
        self.ix = jx
        self.iy = jy
        for i in range(0, self.width):
            inx = i*self.width
            for j in range(0, self.width):
                self.scratch[inx+j] = self.the_image[self.ix+i][self.iy+j]
        return (True, self.the_image[self.ix + self.half][self.iy+self.half], self.scratch)

def main():
    print("Test method for adaptor")

if __name__ == '__main__':
    main()
