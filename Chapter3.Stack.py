# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# The Scientific Python Stack

# <markdowncell>

# The scientific Python stack consists of Numpy, Scipy, and Matplotlib. Numpy brings to Python what Matlab is well known for : strong array manipulation and matrix operations, elementary mathematical functions, random numbers, ...

# <codecell>

import numpy as np

A = np.ones(10)
B = np.arange(5, 10, 0.1) # like range(), but allows non-integer stride
C = np.linspace(0, 2 * np.pi, 30) # between a and b in c steps
D = np.random.normal(0, 1, 20) # also beta, binomial, gamma, poisson, uniform, lognormal, negative binomial, geometric, ...
E = np.sin(C)

print A, "\n"
print B, "\n"
print C, "\n"
print D, "\n"
print E

# <markdowncell>

# Before we go on, a note on *namespaces*. A namespace is a way to "partition" your functions into different spaces. This is particularly useful because you may have functions with the same name, but they may not do the same thing :

# <codecell>

print sin(0.5)
print np.sin(0.5)

sin(0.5) == np.sin(0.5)

# <codecell>

x = [] # x is an empty list ( not a numpy array here )
for i in range(10) :
    x.append(float(i) / 10)

# Equivalent numpy array generation
y = np.arange(0, 1, 0.1)

# These should still be equal
print sin(y), "\n"
print np.sin(y)

# <codecell>

# Numpy arrays can be easily manipulated
blah = np.random.lognormal(0, 1, (5, 3))

print np.dot(blah, blah.T), "\n"
print blah.shape, "\n"
print type(blah)

# <codecell>

# The numpy array is its own type, but it's still a container. It expands on the capabilities of the standard Python list.
# Access is easier though : for multidimensional arrays, you just use a comma per dimension
print type(blah[0, 0])

# <codecell>

# The : operator still works great. Standard arithmetic is ELEMENT-wise. Dimensions are broadcast.
print blah[:, 1] * blah[:, 2] - 2.5

# <codecell>


