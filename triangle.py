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
# don't use many libraries with this

#import numpy as np
#import sys,os

#
# triangle fuzzy numbers
# defined by a midpoint, upper, and lower
#
class triangle:
  def __init__(my, down, mid, up):
    my.centre = mid
    my.low = down
    my.upper = up
# precalculation is important for performance with python
    my.lower_d = mid - down
    my.upper_d = up - mid


  def belief(my, x):
      if x < my.low:
         return 0.0
      if x > my.upper:
         return 0.0
      if x > my.centre:
         return (x-my.centre)/my.upper_d
      else:
         return (my.centre-x)/my.lower_d

# this will associate anything, but
  def associate_rbm(my, an_rbm):
      my.rbm = an_rbm
