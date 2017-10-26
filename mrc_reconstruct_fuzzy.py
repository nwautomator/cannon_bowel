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
# type 2 fuzzy used in this version
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
import triangle

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
# nvis,nhid set the size of the rbm
    hw = 3
    nb = 3
# the number of outer fuzzy sets
    nfuzz = 10
    the_fuzz = []
    delta = 2./(nfuzz-1)
    for i in range(0,nfuzz):
        cent = -1. + 2.*i*delta
        the_fuzz.append( triangle.triangle( cent-delta*2,cent, cent+delta*2))
        the_fuzz[i].associate_rbm( rbm.rbm( (hw*2+1)*(hw*2+1)*nb, 10))
        the_fuzz[i].rbm.its_symmetric()
        the_fuzz[i].rbm.add_fuzzy(-1.,1.,21)
#    the_rbm = rbm.rbm(  (hw*2+1)*(hw*2+1)*nb, 1000)
#    the_rbm.its_symmetric()
#    the_rbm.add_fuzzy(-1.,1., 21)
    the_adapted_image = adapt.adaptor(  hw, 3., the_image)
    the_adapted_image.make_nbit_image(nb)
#    for l in the_rbm.layers:
#        l = np.add(l, the_adapted_image.next(15,15)[2])
#    for l in range(0, len(the_rbm.layers)):
#       the_rbm.layers[l] = np.add( the_rbm.layers[l], the_adapted_image.random_bits()[2])
#    the_adapted_image.reset()
    
    
    print("training starts")
    sys.stdout.flush()
    print("crisp initialization pass")
    sys.stdout.flush() 
    for i in xrange(0,2000):
       a = the_adapted_image.random_bits()
       for f in the_fuzz:
          rate = f.belief(a[1])
          if rate > 0. :
                print(i,a[1],rate,f.rbm.train(a[2],0.1,rate,a[1]))
            
    print("fuzzy pass")
    sys.stdout.flush() 
    for i in xrange(0,2000):
       a = the_adapted_image.random_bits()
       for f in the_fuzz:
          rate = f.belief(a[1])
          if rate > 0. :
                print(i,a[1],rate,f.rbm.train_fuzzy(a[2],0.1,rate,a[1]))
            

    print("training stops")
    sys.stdout.flush()
    
    new_image = np.float32(np.zeros_like(the_image))
#    for i in xrange(0, the_image.shape[0]-2*hw-1):
    for i in xrange(0, 20):
      print("layer", i)
      sys.stdout.flush()
      for j in range(0,the_image.shape[1]-2*hw-1):
#      for j in range(0,20):
          a = the_adapted_image.at_bits(i,j)
          r = the_fuzz[0].rbm
          b = 1.
          fb = the_fuzz[0]
          for f in the_fuzz:
             bt = f.rbm.the_best_layer( a[2])
#             print( bt[1],b)
# catch untrained examples
             if bt[1] < -1.:
                bt[1] = 0.
             if bt[1] < b:
                b = bt[1]
                r = f.rbm
                fb = f
          x = r.estimate_EV( a[2])
          new_image[i][j] = x
          print( x, a[1], fb.centre)
          sys.stdout.flush()
          
#
    the_image = Image.fromarray(np.uint8( rescale(the_adapted_image.the_image,255.)))
    the_image.save('raw.jpg')
    the_image = Image.fromarray(np.uint8( rescale(new_image,255.)))
    the_image.save('rbm.jpg')






main()
