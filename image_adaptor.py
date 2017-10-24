#!/usr/bin/python
# (c) 2017 Treadco software.
# this defaults to python 2 on my machine
#
#

license =''' 
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
import sys,os
from random import random as random


class adaptor:
  def __init__(me, half_width, nsigma, an_array):
    me.width = 1 + 2*half_width  #must be odd x odd to work
# keep this so I can undo it later if I want
    me.mean = an_array.mean()
    me.scale = an_array.std()
    me.nsigma = nsigma
    me.the_image = np.clip( (an_array.__sub__(me.mean)).__div__(float(nsigma)*me.scale), -1., 1.)
    me.scratch = np.float32(np.zeros( (me.width* me.width)))
    me.half = half_width
    me.ix = 0
    me.iy = 0
    me.nx = me.the_image.shape[0]
    me.ny = me.the_image.shape[1]

  def tobits(me,  depth): 
      i = pow(2,depth)     
      me.bits = []
      me.from_bits = []
      for j in range(0,depth):
        me.from_bits.append(pow(2,j)*0.5)
      k = -1.
      for j in range(0,i):
          me.bits.append([k])
          k = -k
# did bit zero by itself because %1
      for l in range(1,depth):
          k = -1.
          sign = 0
          sign_mod = pow(2,l) 
          for j in range(0,i):
             me.bits[j].append(k)
             sign = sign + 1
             if sign%sign_mod == 0:
                 k = -k
      me.range_delta = float(i-1) #avoid picket fence
      me.nbits = depth

  def bits_to_density(me, bits):
     ac = 0.
     for i in range(0,me.nbits):
         ac = ac + me.from_bits[i]*(bits[i] + 1.)
     return ac

  def make_nbit_image(me, depth):
     me.tobits( depth)
     me.scratch = np.float32(np.zeros( (me.width* me.width*depth)))
     ma = me.the_image.max()
     mi = me.the_image.min()
     me.indexes = np.uint8(np.zeros_like(me.the_image))
     for i in range(0,me.nx):
        for j in range(0,me.ny):
           me.indexes[i][j] = int( (me.the_image[i][j]-mi)/(ma-mi)*me.range_delta)
#           print( me.indexes[i][j], (me.the_image[i][j]-mi)/(ma-mi),me.the_image[i][j])



  def to_original(me, apixel):
    return apixel*me.scale*(float(me.sigma)) + me.mean

  def reset(me):
      me.ix = 0
      me.iy = 0

  def random_bits(me):
    rx = int(random()*(me.nx - me.width))
    ry = int(random()*(me.ny - me.width))
    for i in range(0, me.width):
      inx = i*me.width*me.nbits
      for j in range(0, me.width):
        inj = j*me.nbits
        for k in range(0,me.nbits):
          me.scratch[inx +inj + k ] = (me.bits[me.indexes[rx+i][ry+j]])[k]
    return (True, me.the_image[ rx + me.half][ry+me.half], me.scratch) 
   

  def random(me):
    rx = int(random()*(me.nx - me.width))
    ry = int(random()*(me.ny - me.width))
    for i in range(0, me.width):
      inx = i*me.width
      for j in range(0, me.width):
        me.scratch[inx +j] = me.the_image[rx+i][ry+j]
    return (True, me.the_image[ rx + me.half][ry+me.half], me.scratch) 

  def at_bits(me,rx,ry):
    for i in range(0, me.width):
      inx = i*me.width*me.nbits
      for j in range(0, me.width):
        inj = j*me.nbits
        for k in range(0,me.nbits):
          me.scratch[inx +inj + k ] = (me.bits[me.indexes[rx+i][ry+j]])[k]
    return (True, me.the_image[ rx + me.half][ry+me.half], me.scratch) 
   

  def at(me,rx,ry):
    for i in range(0, me.width):
      inx = i*me.width
      for j in range(0, me.width):
        me.scratch[inx +j] = me.the_image[rx+i][ry+j]
    return (True, me.the_image[ rx + me.half][ry+me.half], me.scratch) 
   

  def next(me, dx,dy=0):
    jx = me.ix +dx
    jy = me.iy
    if dy == 0 :
      if jx > me.nx - me.width:
          jy = me.iy + 1
          jx = 0
    else:
       jy = me.iy + dy
    if jy >= me.ny - me.width or jx >= me.nx - me.width:
       return (False, 0.,me.scratch)  
    me.ix = jx
    me.iy = jy
    for i in range(0, me.width):
      inx = i*me.width
      for j in range(0, me.width):
        me.scratch[inx +j] = me.the_image[me.ix+i][me.iy+j]
    return (True, me.the_image[ me.ix + me.half][me.iy+me.half], me.scratch) 


#def main():
#   print("Test method for adaptor")

#main()
