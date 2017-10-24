#!/usr/bin/python
# this defaults to python 2 on my machine
# (c) 2017 Treadco software.
#
# Regularized image sharpening library
#
# MRC file version
#
# rbm is trained on image, then reconstruction is used to generate a new one.
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
    hw = 5
    nb = 1
    the_rbm = rbm.rbm(  (hw*2+1)*(hw*2+1)*nb, 1000)
    the_rbm.its_symmetric()
    the_adapted_image = adapt.adaptor(  hw, 3., the_image)
    the_adapted_image.make_nbit_image(nb)
#    for l in the_rbm.layers:
#        l = np.add(l, the_adapted_image.next(15,15)[2])
    for l in range(0, len(the_rbm.layers)):
       the_rbm.layers[l] = np.add( the_rbm.layers[l], the_adapted_image.random_bits()[2])
    the_adapted_image.reset()
    
    
    print("training starts")
    sys.stdout.flush()
  #  a = the_adapted_image.next(1)
    for i in xrange(0,2000):
       a = the_adapted_image.random_bits()
       print(i,a[1],the_rbm.train(a[2],0.1,1.0,a[1]))
#       if i > 1000:
#        print a[2]
#        print the_rbm.reconstruct(a[2])
#       print(i, a[1])

    print("training stops")
    sys.stdout.flush()
    
    new_image = np.float32(np.zeros_like(the_image))
    for i in xrange(0, the_image.shape[0]-2*hw-1):
#    for i in xrange(0, 20):
      print("layer", i)
      sys.stdout.flush()
#      for j in range(0,the_image.shape[1]-2*hw-1):
      for j in range(0,20):
          a = the_adapted_image.at_bits(i,j)
          b = the_rbm.reconstruct(a[2])
          for k in range(0,2*hw+1):
            ink = k *(2*hw+1)*nb
            for l in range(0,2*hw+1):
                new_image[i+k][j+l] += the_adapted_image.bits_to_density( b[ink+l*nb:ink+l*nb+nb])
                #print the_adapted_image.bits_to_density( b[ink+l*nb:ink+l*nb+nb])
                 #sys.stdout.flush()
#    for layer in layers:
#        the_image = np.add(the_image,layer)
    the_image = Image.fromarray(np.uint8( rescale(the_adapted_image.the_image,255.)))
    the_image.save('raw.jpg')
    the_image = Image.fromarray(np.uint8( rescale(new_image,255.)))
    the_image.save('rbm.jpg')






main()
