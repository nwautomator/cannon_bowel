#!/usr/bin/python
# this defaults to python 2 on my machine
# (c) 2017 Treadco software.
#
# python version of the fuzzy rbm
# supports the non-fuzzy version.
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



#
# use pickle.dump(instance,file)
#  and pickle.load(file)
#
# to save and restore data. file is a Python FILE object so 
# it's opened first.
#
#


#
#  fuzzy uses an ndarray so the ndarray primitives are available
#  we'll wrap some of them into a standard interface, but you can
#  roll your own if need be.
#


class fuzzy:
  def __init__(me, the_min,the_max,the_number_of_divisions):
    me.my_min = the_min
    me.my_max = the_max
    me.delta = (the_max-the_min)/the_number_of_divisions
    me.nd = the_number_of_divisions
    me.counts = np.float32(np.zeros(the_number_of_divisions))
    ddi = the_number_of_divisions/2
    me.args = []
    for i in range(0, the_number_of_divisions):
      me.args.append((i-ddi)*me.delta)

  def initialize_counts(me):
    me.counts.__imul__(0.)

  def add(me, what):
    i = int(( what - me.my_min)/me.delta +0.5)
#    print(what,i)
# insert rangechecking here.
    if i >= me.nd:
      i = me.nd -1
    if i < 0:
      i = 0
    me.counts[i] += 1.
#    print(what,me.counts[i],i)
#    sys.stdout.flush()


  def expected_value(me):
    ds = me.counts.sum()
    if ds == 0.:
       ds = 1.
    dsum = np.dot( me.args, me.counts)
    return dsum/ds  
#       print me.counts
#       return (me.my_min + me.my_max)*0.5
#    dsum = 0.
#    ddi = len(me.counts)/2
#    for i in range(0, len(me.counts)):
#       dx = (i-ddi) *me.delta
#       dsum += me.counts[i]*dx
#    return dsum/ds  
# use numpy you dumb fsck
#    im = np.argmax(me.counts)
#    return float(im)*me.delta + me.my_min
       
  def belief(me):
    ds = me.counts.sum()
    if ds == 0.:
       return (me.my_min + me.my_max)*0.5,0.
# use numpy you dumb fsck
    im = float(np.argmax(me.counts))
    return (im*me.delta + me.my_min),im/ds
       
  def damp(me, avalue):
    ds = me.counts.sum()
    if ds == 0.:
       return 1.
    im = int((avalue - me.my_min)/(me.my_max-me.my_min)+0.5)
    if im >= me.nd:
      im = me.nd -1
    if im < 0:
      im = 0
    i = np.argmax(me.counts)
    if abs(i-im) < 2:
       return 1.
    return -0.1 

def main():
  print("this is the main routine, defined for testing purposes")

  simon = fuzzy(-1.,1., 10)
  simon.add(0.)
  simon.add(0.1)
  simon.add(0.2)
  simon.add(0.3)
  print( simon.counts)
  print( simon.expected_value()) 
  simon.initialize_counts()
  print( simon.counts)
  simon.add(0.)
  simon.add(0.1)
  simon.add(0.2)
  print( simon.counts)
  print( simon.expected_value()) 
  

#main()
