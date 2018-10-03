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

#
# triangle fuzzy numbers
# defined by a midpoint, upper, and lower
#

class Triangle:
    def __init__(self, down, mid, up):
        self.centre = mid
        self.low = down
        self.upper = up
        # precalculation is important for performance with python
        self.lower_d = mid - down
        self.upper_d = up - mid

    def belief(self, x):
        if x < self.low:
            return 0.0
        if x > self.upper:
            return 0.0
        if x > self.centre:
            return 1.-(x-self.centre)/self.upper_d
        else:
            return 1.-(self.centre-x)/self.lower_d

# this will associate anything, but
    def associate_rbm(self, an_rbm):
        self.rbm = an_rbm
