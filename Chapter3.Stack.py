# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# The Scientific Python Stack

# <headingcell level=3>

# Numpy

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

# Another function for sin can be found in the Python math library
import math

print math.sin(0.5)
print np.sin(0.5)

sin(0.5) == np.sin(0.5)

# <codecell>

# BUT
x = np.linspace(0, 2 * np.pi, 30)
print np.sin(x), "\n"
print sin(x)

# <markdowncell>

# Note how this function doesn't like to be passed lists or arrays. It's thrown us an error, and provided us with a **traceback** - a "chronological" trace of where the error occurred, and what caused it. This one's not particularly long, as the error is immediate ( and not in a subfunction ), but it does tell us which line it's on. It seems that `math.sin()` doesn't like to be passed anything but length-1 arrays - that is, single floats or ints. Not particularly useful when we're trying to be array-oriented, which is what numpy excels at.
# 
# OK, back to numpy.

# <codecell>

# Numpy arrays can be easily manipulated
blah = np.random.lognormal(0, 1, (4, 3))

print np.dot(blah, blah.T), "\n"
print blah.shape, "\n"
print type(blah)

# <codecell>

# The numpy array is its own type, but it's still a container. It expands on the capabilities of the standard Python list.
# Access is easier though : for multidimensional arrays, you just use a comma per dimension
print type(blah[0, 0])
print blah[:, 2::2]

# <codecell>

# The : operator is still used. Standard arithmetic is ELEMENT-wise. Dimensions are broadcast.
print blah[:, 1] * blah[:, 2] - 2.5
print blah.sum(), blah.mean(), blah.min(), blah.max(), blah.var(), blah.cumsum().std()

# <codecell>

# Quick note on interactive environments. If you're in IPython, Spyder, or similar, you can tab-complete :
blah

# <codecell>

# We can manipulate the shapes of arrays
print blah.shape
print blah.ravel().shape # Unravel / flatten the array
print blah.reshape(2, 6).shape
print blah.reshape(2, 6).T

# <markdowncell>

# **Linear Algebra**
# 
# Numpy has a module called `linalg`, which offers things like norms, decompositions, matrix powers, inner and outer products, trace, eigenvalues, etc...

# <codecell>

# Linear algebra
A = np.random.normal(size=(3, 3))
print np.linalg.inv(A), "\n"

# Modules can be imported under their own namespace for short
import numpy.linalg as lin
print lin.inv(A), "\n"
print lin.eigvals(A), "\n"

# If we want to do some real linear algebra, we can skip elementwise syntax by declaring an array as a numpy matrix :
B = np.matrix(A)
print B * B, "\n" # now MATRIX multiplication
print A * A # still ELEMENTWISE multiplication

# <headingcell level=3>

# Matplotlib

# <markdowncell>

# Before we continue onto Scipy, let's look into some plotting. Matplotlib is aimed to feel familiar to Matlab users. 

# <codecell>

import matplotlib.pyplot as plt
figsize(12, 8)

x = np.random.exponential(3, size=10000)

plt.subplot(1, 2, 1)
plt.plot(x[::100])
plt.subplot(1, 2, 2)
plt.hist(x, 50)
plt.show()

# Note that, in an interactive setting, you will need to call plt.show(), which I've included above for completeness
# In IPython using the --pylab inline argument, you don't need it; I'll drop it from here 

# <codecell>

# Like Matlab, we can feed style arguments to the plot function
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + np.random.normal(scale = 0.2, size = 100)

# Calling plot() several times without creating a new figure or subplot puts them on the same figure
plt.plot(x, y, "r.")
plt.plot(x, np.sin(x), "k-")

# <codecell>

# 50x50x3 matrix on a scatter plot, third dimension plotted as size
x = np.random.uniform(low = 0, high = 1, size = (50, 50, 3))

# Transparency ? Sure.
plt.scatter(x[:, 0], x[:, 1], s=x[:, 2]*500, alpha = 0.3)

# Set axis limits
plt.xlim([0, 1])
plt.ylim([0, 1])

# <codecell>

# Area plots ?
x = np.linspace(0, 6 * np.pi, 100)
y = np.exp(-0.2*x) * np.sin(x)

plt.plot(x, y, "k", linewidth=4)
plt.fill_between(x, y, y2=0, facecolor="red")

# <codecell>

# Error bars ? Let's say we got ten measurements of the above process, the decaying sine wave
print y.shape
measuredy = np.tile(y, (100, 1)) # tile y, 10x1 copies
print measuredy.shape

# Let's add some heteroscedastic noise to our observations
measuredy = np.random.normal(loc=y, scale=np.abs(y+0.01)/2., size = measuredy.shape)

# Let's assume we already know our error is Gaussian, for simplicity. Compute mean and std :
estmean = measuredy.mean(axis=0)
eststd = measuredy.std(axis=0)

# Plot the estimated mean with one standard deviation error bars, and the real signal
plt.plot(x, y, "r", linewidth=3)
plt.errorbar(x, estmean, yerr = eststd)

# <codecell>

# Two dimensional, or image plots ? Perlin noise is fun !

# Initial 2D noisy signal
X = np.random.normal(size=(256, 256))
plt.figure()
plt.imshow(X, cmap="gray")
plt.title("Original 2D Gaussian noise")

# We'll use a 2D Gaussian for smoothing, with identity as covariance matrix
# We'll grab a scipy function for it for now
import scipy.ndimage as nd
plt.figure()
temp = np.zeros_like(X)
temp[128, 128] = 1.
plt.imshow(nd.filters.gaussian_filter(temp, 20, mode="wrap"))
plt.title("Gaussian kernel")

# Generate the Perlin noise
perlin = zeros_like(X)
for i in 2**np.arange(6) :
    perlin += nd.filters.gaussian_filter(X, i, mode="wrap") * i**2
    
# and plot it several ways
plt.figure()
plt.imshow(perlin)
plt.title("Perlin Noise")

plt.figure()
plt.imshow(perlin, cmap="gray")
plt.contour(perlin, linewidths=3)
plt.title("Greyscale, with contours")
#plt.xlim([0, 256])
#plt.ylim([0, 256])
#plt.axes().set_aspect('equal', 'datalim')

# <codecell>

# And, of course ( stolen from Jake VanderPlas )

def norm(x, x0, sigma):
    return np.exp(-0.5 * (x - x0) ** 2 / sigma ** 2)

def sigmoid(x, x0, alpha):
    return 1. / (1. + np.exp(- (x - x0) / alpha))
    
# define the curves
x = np.linspace(0, 1, 100)
y1 = np.sqrt(norm(x, 0.7, 0.05)) + 0.2 * (1.5 - sigmoid(x, 0.8, 0.05))

y2 = 0.2 * norm(x, 0.5, 0.2) + np.sqrt(norm(x, 0.6, 0.05)) + 0.1 * (1 - sigmoid(x, 0.75, 0.05))

y3 = 0.05 + 1.4 * norm(x, 0.85, 0.08)
y3[x > 0.85] = 0.05 + 1.4 * norm(x[x > 0.85], 0.85, 0.3)


with plt.xkcd() :
    plt.plot(x, y1, c='gray')
    plt.plot(x, y2, c='blue')
    plt.plot(x, y3, c='red')
    
    plt.text(0.3, -0.1, "Yard")
    plt.text(0.5, -0.1, "Steps")
    plt.text(0.7, -0.1, "Door")
    plt.text(0.9, -0.1, "Inside")
    
    plt.text(0.05, 1.1, "fear that\nthere's\nsomething\nbehind me")
    plt.plot([0.15, 0.2], [1.0, 0.2], '-k', lw=0.5)
    
    plt.text(0.25, 0.8, "forward\nspeed")
    plt.plot([0.32, 0.35], [0.75, 0.35], '-k', lw=0.5)
    
    plt.text(0.9, 0.4, "embarrassment")
    plt.plot([1.0, 0.8], [0.55, 1.05], '-k', lw=0.5)
    
    plt.title("Walking back to my\nfront door at night:")
    
    plt.xlim([0, 1])
    plt.ylim([0, 1.5])

# <headingcell level=3>

# Scipy

# <markdowncell>

# Scipy does all sorts of awesome stuff : integration, interpolation, optimisation, FFTs, signal processing, more linear algebra, stats, image processing, ...

# <codecell>

import scipy as sp
import scipy.spatial as spatial
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.signal import detrend
import scipy.stats as st

# Convex hulls
ax = plt.axes()
points = np.random.normal(0.5, 0.2, size=(20,2))
plt.scatter(points[:, 0], points[:, 1], s=200, c="r")
hull = spatial.ConvexHull(points)
for i in hull.simplices :
    plt.plot(points[i, 0], points[i, 1], "k", linewidth=4)
spatial.voronoi_plot_2d(spatial.Voronoi(points), ax)
plt.title("Convex Hull and Voronoi Tessellation")


# Ordinary Differential Equations
r0 = 16.
gamma = .5
t = np.linspace(0, 8, 200)
def SIR(y, t) :
    S = - r0 * gamma * y[0] * y[1]
    I = r0 * gamma * y[0] * y[1] - gamma * y[1]
    R = gamma * y[1]
    return [S, I, R]
solution = odeint(SIR, [0.99, .01, 0.], t)

plt.figure()
plt.plot(t, solution, linewidth=3)
plt.title("Measles, Single Epidemic")
plt.xlabel("Time (weeks)")
plt.ylabel("Proportion of Population")
plt.legend(["Susceptible", "Infected", "Recovered"])


# Signal Processing - create trending signal, detrend, FFT
x = np.linspace(0, 1, 300)
trend = np.zeros_like(x)
trend[100:200] = np.linspace(0, 10, 100)
trend[200:] = np.linspace(10, -20, 100)
y = np.random.normal(loc = 3*np.sin(2*np.pi*20*x)) + trend # Signal is a sine wave with noise and trend
yt = detrend(y, bp = [100, 200]) # detrend, giving break points
Yfft = sp.fftpack.rfft(yt) # Calculate FFT
freqs = sp.fftpack.rfftfreq(len(yt), x[1] - x[0]) # Frequencies of the FFT

plt.figure()
plt.subplot(311)
plt.title("Detrending")
plt.plot(x, y, linewidth=2)
plt.plot(x, trend, linewidth=4)
plt.subplot(312)
plt.plot(x, yt, linewidth=2)
plt.subplot(313)
plt.title("FFT")
plt.plot(freqs, (np.abs(Yfft)**2), linewidth=2)


# Distributions
plt.figure()
plt.subplot(131)
plt.title("Normal PDF")
plt.plot(np.arange(-5, 5, 0.1), st.norm.pdf(np.arange(-5, 5, 0.1), 0, 1), linewidth=2)
plt.subplot(132)
plt.title("Beta CDF")
plt.plot(st.beta.cdf(np.linspace(0, 1, 100), 0.5, 0.5), linewidth=2)
plt.subplot(133)
plt.title("Erlang PDF")
plt.plot(np.linspace(0, 10, 100), st.erlang.pdf(np.linspace(0, 10, 100), 2, loc=0), linewidth=2)


# Statistical Tests
a = np.random.normal(0, 1.5, size=1000)
b = np.random.normal(.2, 1, size=1000)
plt.figure()
plt.hist(a, 30)
plt.hist(b, 30)
plt.title("p = " + str(st.ttest_ind(a, b)[1]))

# <markdowncell>

# There's a lot more you can do with `numpy`, `scipy`, and `matplotlib`. The documentation's pretty good, so look around for what you're after - it's probably already been implemented.

