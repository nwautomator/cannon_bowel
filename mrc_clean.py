#!/usr/bin/python
# this defaults to python 2 on my machine
# (c) 2017 Treadco software.
#
# Regularized image sharpening library
#
# MRC file version
# 
license = ''' Copyright (c) 2017  Treadco LLC, Amelia Treader, Robert W Harrison
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
import sys,os
from PIL import Image
from PIL import ImageChops
#from pylab import *
import mrcfile
import rbm
import fuzzy
import image_adaptor as adapt


def rescale(a, upper):
   amax = a.max()
   amin = a.min()
   amax -= amin
   return (a.__sub__(amin)).__mul__(upper/amax)
#   b = a.__sub__(amin)
#   c = b.__mul__(upper/amax)
#   return c



def main():
    try:
      original = mrcfile.open(sys.argv[1],mode='r')
    except IOError:
      print("Could not open the input \nUsage tick_mrc inputfile outputfile.")
      sys.exit()

# create list of layers.
    layers = []
    for layer in original.data:
       layers.append(np.float32(layer))
#layers are the arrays containing the data.
    the_image = np.zeros_like(layers[0])
    the_image = np.add(the_image, layers[40])
# nvis,nhid
    the_rbm = rbm.rbm(  (15*2+1)*(15*2+1), 100)
    the_rbm.add_fuzzy(-1., 1., 21)
    the_rbm.reinitialize_fuzzy()
    the_adapted_image = adapt.adaptor(  15, 3., the_image)
#    for l in the_rbm.layers:
#        l = np.add(l, the_adapted_image.next(15,15)[2])
#    for l in range(0, len(the_rbm.layers)):
#       the_rbm.layers[l] = np.add( the_rbm.layers[l], the_adapted_image.next(1,1)[2])
    the_adapted_image.reset()
    
    print("training starts")
    sys.stdout.flush()
  #  a = the_adapted_image.next(1)
    for i in range(0,1000):
       a = the_adapted_image.random()
       the_rbm.train(a[2],0.1,0.5,a[1])
       print(i, a[1])

    a = the_adapted_image.random()
    i = 0
    j = 0
    errs = np.float32(np.zeros(100))
    while( a[0] ):
        the_rbm.train_fuzzy(a[2],0.1,0.5,a[1])
        if i == 1000:
           the_rbm.reinitialize_fuzzy()
        errs[j] += a[1]-the_rbm.estimate_EV(a[2])
        j = (j+1)%100
        print(i,a[1], errs.std())
#        print(i, a[1], the_rbm.estimate_EV(a[2]))
#        a = the_adapted_image.next(1,1)
        a = the_adapted_image.random()
        sys.stdout.flush()
        i += 1
    print("training stops")
    sys.stdout.flush()
    

#    for layer in layers:
#        the_image = np.add(the_image,layer)
    the_image = Image.fromarray(np.uint8( rescale(the_adapted_image.the_image,255.)))
    the_image.save('sum.jpg')






main()
