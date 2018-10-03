''' Copyright (c) 2017  Treadco LLC, Amelia Treader, Robert W Harrison
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

# Regularized image sharpening library
# MRC file version

import numpy as np
import sys
import os
from PIL import Image
from PIL import ImageChops
import mrcfile
from cannon_bowel import rbm
from cannon_bowel import fuzzy
import image_adaptor as adapt

def rescale(a, upper):
    amax = a.max()
    amin = a.min()
    amax -= amin
    return (a.__sub__(amin)).__mul__(upper/amax)

def main():
    try:
        original = mrcfile.open(sys.argv[1], mode='r')
    except IOError:
        print("Could not open the input \nUsage tick_mrc inputfile outputfile.")
        sys.exit()

    # create list of layers.
    layers = []
    for layer in original.data:
        layers.append(np.float32(layer))
    # layers are the arrays containing the data.
    the_image = np.zeros_like(layers[0])
    the_image = np.add(the_image, layers[40])
    # nvis,nhid
    hw = 1
    the_rbm = rbm.RBM((hw*2+1)*(hw*2+1), 10)
    the_rbm.add_fuzzy(-1., 1., 21)
    the_rbm.reinitialize_fuzzy()
    the_adapted_image = adapt.Adaptor(hw, 3., the_image)
    the_adapted_image.reset()

    print("training starts")
    sys.stdout.flush()
    for i in range(0, 1000):
        a = the_adapted_image.random()
        the_rbm.train(a[2], 0.1, 0.5, a[1])
        print(i, a[1])

    a = the_adapted_image.random()
    i = 0
    j = 0
    errs = np.float32(np.zeros(100))
    while(a[0]):
        the_rbm.train_fuzzy(a[2], 0.1, 0.5, a[1])
        if i == 1000:
            the_rbm.reinitialize_fuzzy()
        x = the_rbm.estimate_expected_value(a[2])
        errs[j] = a[1]-x
        j = (j+1) % 100
        print(i, a[1], x,  errs.std())
        a = the_adapted_image.random()
        sys.stdout.flush()
        i += 1
    print("training stops")
    sys.stdout.flush()
    the_image = Image.fromarray(
        np.uint8(rescale(the_adapted_image.the_image, 255.)))
    the_image.save('sum.jpg')

if __name__ == '__main__':
    main()
