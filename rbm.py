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
from math import exp as exp



#
# use pickle.dump(instance,file)
#  and pickle.load(file)
#
# to save and restore data. file is a Python FILE object so 
# it's opened first.
#
#

class rbm:  #the basic rbm
  def __init__(me, number_visible, number_hidden):
    me.nvis = number_visible 
    me.nhid = number_hidden
    me.layers = []
    me.energies = []
    me.hidden = []
# initialize the space
# making essentially empty lists means that we can avoid using append etc
    me.scratch = np.full(number_visible,0.0)
    for i in range(0,number_hidden):
        me.layers.append(np.float32(np.zeros(number_visible)))
        me.hidden.append(0)
        me.energies.append(0.) 


  def reconstruct(me, data, use_best = True):
    the_layer = me.the_best_layer(data) 
    ib = the_layer[0]
    a = me.layers[ib]
    sign = 1.
    if me.hidden[ib] < 0.:
       sign = -1.
#
# there may be a clever numpy solution for this loop
#
    for i in range(0,me.nvis):
       me.scratch[i] = 1.
       if a[i] < 0.:
           me.scratch[i] = -1.
    return me.scratch.__mul__(sign) 



  def the_best_layer(me, data, use_best = True):
    if use_best:
      me.assign_hidden_and_reconstruction_energy(data)
    else:
      me.assign_hidden_and_energy(data)
    ib = 0
    eb = me.energies[0]
    for i in range(1,me.nhid):
       if me.energies[i] < eb:
          ib = i
          eb = me.energies[i]
    return ib,eb


  def assign_hidden_and_reconstruction_energy(me, data):
    for i in range(0, me.nhid):
       eraw = np.dot( data, me.layers[i])
       ebest = np.dot( data.__abs__(), (me.layers[i]).__abs__())
       if ebest == 0.0:
          ebest = 1.0
       if eraw > 0.:
          me.hidden[i] = -1.0
          me.energies[i] = -eraw/ebest
       else:
          me.hidden[i] = 1.0
          me.energies[i] = eraw/ebest


  def assign_hidden_and_energy(me, data):
    for i in range(0, me.nhid):
       eraw = np.dot( data, me.layers[i])
       if eraw > 0.:
          me.hidden[i] = -1.0
          me.energies[i] = -eraw
       else:
          me.hidden[i] = 1.0
          me.energies[i] = eraw

  def trainOmatic(me,data,beta,learning_rate,use_best = True):
      me.train(data,beta,learning_rate,use_best)
      me.antitrain(data,beta,learning_rate*0.1,use_best)


  def train(me,data,beta,learning_rate, use_best = True):
    if use_best:
      me.assign_hidden_and_reconstruction_energy(data)
    else:
      me.assign_hidden_and_energy(data)
# select the row to train.
    imin = 0
    emin = me.energies[0]
    for i in range(1,me.nhid):
      if emin > me.energies[i] :
         imin = i
         emin = me.energies[i]
#
# emin,imin now point to the best row
#    
    hsign = me.hidden[imin]
    alayer = me.layers[imin]
# the products with hsign keep the +- straight.
# for the gradients that is.
    for i in range(0, me.nvis): # over the row
      ef = alayer[i]*hsign*data[i]
      ep = ef*beta*hsign
      em = -ep
      fp = exp(-ep)
      fm = exp(-em)
      damp = (fp-fm)/(fp+fm)  *hsign  *data[i]
      hv = hsign *data[i]
      alayer[i] += learning_rate*( -hv + damp)


  def antitrain(me,data,beta,learning_rate,use_best = True):
    if use_best:
      me.assign_hidden_and_reconstruction_energy(data)
    else:
      me.assign_hidden_and_energy(data)
# select the row to train.
    imax = 0
    emax = me.energies[0]
    for i in range(1,me.nhid):
      if emax <= me.energies[i] :
         imax = i
         emax = me.energies[i]
#
# emin,imin now point to the best row
#    
    hsign = me.hidden[imax]
    alayer = me.layers[imax]
# the products with hsign keep the +- straight.
# for the gradients that is.
    for i in range(0, me.nvis): # over the row
      ef = alayer[i]*hsign*data[i]
      ep = ef*beta*hsign
      em = -ep
      fp = exp(-ep)
      fm = exp(-em)
      damp = (fp-fm)/(fp+fm)  *hsign  *data[i]
      hv = hsign *data[i]
      alayer[i] -= learning_rate*( -hv + damp)




def main():
   print("this is the main routine, set up for testing")
   my_rbm = rbm(2,2)
   print(my_rbm.layers)   
   d = np.full(2,1.)
   d[0] = -1.0
   my_rbm.trainOmatic(d, 0.1, 0.1)
   print(my_rbm.layers)   
   for i in range(1,10):
     my_rbm.train(d, 0.1, 0.1)
     print(my_rbm.layers)   

   d[0] = 0.
   print(my_rbm.reconstruct(d))
   d[0] = 1.
   d[1] = 0.
   print(my_rbm.reconstruct(d))


main()