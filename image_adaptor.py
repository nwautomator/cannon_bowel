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


class adaptor:
  def __init__(me, half_width, nsigma, an_array):
    me.width = 1 + 2*half_width  #must be odd x odd to work
# keep this so I can undo it later if I want
    me.mean = an_array.mean()
    me.scale = an_array.std()
    me.nsigma = nsigma
    me.the_image = np.clip( (an_array.__sub__(me.mean)).__mul__(float(nsigma)/me.scale), -1., 1.)
    me.scratch = np.float32(np.zeros( (me.width* me.width)))
    me.half = half_width
    me.ix = 0
    me.iy = 0
    me.nx = me.the_image.shape[0]
    me.ny = me.the_image.shape[1]

  def to_original(me, apixel):
    return apixel*me.scale/(float(me.sigma)) + me.mean

  def reset(me):
      me.ix = 0
      me.iy = 0

  def next(me, dx,dy=0):
    jx = me.ix +dx
    if dy == 0 :
      if jx > me.nx - me.width:
          jy = me.iy + 1
          jx = 0
    else:
       jy = me.iy + dy
    if jy > me.ny - me.width or jy > me.nx - me.width:
       return (False, 0.,me.scratch)  
    me.ix = jx
    me.iy = jy
    for i in range(0, me.width):
      inx = i*me.width
      for j in range(0, me.width):
        me.scratch[inx +j] = me.the_image[me.ix+i][me.iy+j]
    return (True, me.the_image[ me.ix + me.half][me.iy+me.half], scratch) 


def main():
   print("Test method for adaptor")

main()
