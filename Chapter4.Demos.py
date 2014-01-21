# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Demos and Cool Stuff

# <codecell>

import numpy as np
from scipy.integrate import odeint
import scipy.stats as st
import scipy.spatial as sp
import scipy.ndimage as nd
import scikits.bootstrap as bootstrap
import matplotlib.pyplot as plt
import seaborn
import lmfit as lm
import pymc
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets, svm
from sklearn.cross_validation import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, morphology, exposure, filter, feature, color, measure
import pickle
figsize(12, 6)

# <headingcell level=3>

# LMFit : Nonlinear, Constrained Minimisation

# <codecell>

# http://lmfit.github.io/lmfit-py/parameters.html#simple-example
# Let's create a damped sinusoid
# True function :
# 5 sin(2x + 0.3) * exp(-0.03*(x^2))
x = np.linspace(0, 15, 300)
y = 5 * np.sin(2 * x + 0.3) * np.exp(-0.03 * (x**2)) + np.random.normal(size=len(x), scale=0.4)
plt.plot(x, y)

# <codecell>

# We can use lmfit to fit this function to a model
def objectivefunction(params, x, data) :
    amplitude = params["amplitude"].value
    phase = params["phase"].value
    frequency = params["frequency"].value
    decay = params["decay"].value
    power = params["power"].value
    
    model = amplitude * np.sin(frequency * x + phase) * np.exp(decay * (x**power))
    return model - data


# We create a parameters structure
params = lm.Parameters()
params.add("amplitude", value = 10, min = 0)
params.add("phase", value = 0, min = -np.pi, max = np.pi)
params.add("frequency", value = 1)
params.add("decay", value = -1, max = 0)
params.add("power", value = 1, min = 0)

# Fit the function
result = lm.minimize(objectivefunction, params, args = (x, y))

plt.plot(x, y + result.residual, linewidth=3)
plt.plot(x, y, "o")

lm.report_fit(params)

# <headingcell level=3>

# PyMC : MCMC with No U-Turn Sampler

# <codecell>

"""
# Let's infer R0 and gamma in an SIR model
t = np.linspace(0, 8, 100)

def SIR(y, t, r0, gamma) :
    S = - r0 * gamma * y[0] * y[1]
    I = r0 * gamma * y[0] * y[1] - gamma * y[1]
    R = gamma * y[1]
    return [S, I, R]

solution = odeint(SIR, [0.99, 0.01, 0], t, args=(16., 0.5))

with pymc.Model() as model :
    r0 = pymc.Normal("r0", 15, sd=10)
    gamma = pymc.Uniform("gamma", 0.3, 1.)
    
    # Use forward Euler to solve
    dt = t[1] - t[0]

    S = [0.99]
    I = [0.01]
    
    for i in range(1, len(t)) :
        S.append(pymc.Normal("S%i" % i, \
                             mu = S[-1] + dt * (-r0 * gamma * S[-1] * I[-1]), \
                             sd = solution[:, 0].std()))
        I.append(pymc.Normal("I%i" % i, \
                             mu = I[-1] + dt * ( r0 * gamma * S[-1] * I[-1] - gamma * I[-1]), \
                             sd = solution[:, 1].std()))

    Imcmc = pymc.Normal("Imcmc", mu = I, sd = solution[:, 1].std(), observed = solution[:, 1])

    #start = pymc.find_MAP()
    trace = pymc.sample(2000, pymc.NUTS())
"""

# <codecell>

#pymc.traceplot(trace);

# <headingcell level=3>

# Pickle - Saving Python Objects to Disk

# <codecell>

# Create a dictionary of random objects
x = { "Species": "Homo sapiens", "Age": 73, "Foods" : ["Chocolate", "Coffee", "Waffles"] }

# Let's save this arbitrary object to file
pickle.dump(x, open("pickled_human.p", "w"))

# Now let's read it back in
human = pickle.load(open("pickled_human.p", "r"))

# We can access it as normal
print human["Foods"]

# <headingcell level=3>

# Pandas - Data Management

# <codecell>

# http://www.randalolson.com/2012/08/06/statistical-analysis-made-easy-in-python/
# Thanks to Luis Zaman for the data
parasites = pd.read_csv("parasite_data.csv", na_values = [" "])
print parasites

# <codecell>

# Access by column names
print "Mean virulence : %f" % parasites["Virulence"].mean()

# Let's drop N/A values
parasites = parasites.dropna()
print parasites

# <codecell>

# Let's look at the data for three given virulences
p1 = parasites[parasites["Virulence"] == 0.5]["ShannonDiversity"]
p2 = parasites[parasites["Virulence"] == 0.8]["ShannonDiversity"]
p3 = parasites[parasites["Virulence"] == 0.9]["ShannonDiversity"]

plt.subplot(321)
plt.title("Vir = 0.5")
plt.plot(p1, linewidth=3)
plt.axhline(p1.mean(), color="r", linewidth=3)

plt.subplot(323)
plt.title("Vir = 0.8")
plt.plot(p2, linewidth=3)
plt.axhline(p2.mean(), color="r", linewidth=3)

plt.subplot(325)
plt.title("Vir = 0.9")
plt.plot(p3, linewidth=3)
plt.axhline(p3.mean(), color="r", linewidth=3)

plt.subplot(322)
plt.hist(p1)
plt.axvline(p1.mean(), color="r", linewidth=3)

plt.subplot(324)
plt.hist(p2)
plt.axvline(p2.mean(), color="r", linewidth=3)

plt.subplot(326)
plt.hist(p3)
plt.axvline(p3.mean(), color="r", linewidth=3)

plt.tight_layout()

# <codecell>

# What about comparing two different virulences ?
print "p-value for Mann-Whitney test between 1 and 2 : %f" % st.ranksums(p1, p2)[1]
print "p-value for Mann-Whitney test between 1 and 3 : %f" % st.ranksums(p1, p3)[1]
print "p-value for Mann-Whitney test between 2 and 3 : %f" % st.ranksums(p2, p3)[1]

# <codecell>

# That's a bad idea, statistically. We could go with ANOVA :
print "One way ANOVA, p-value : %f" % st.f_oneway(p1, p2, p3)[1]

# <codecell>

# We can boostrap confidence intervals and plot those, too - using scikits.bootrap
CIs = bootstrap.ci(p1, n_samples=50000)
downerror = np.array(p1 - CIs[0])
uperror = np.array(p1 + CIs[1])

plt.plot(p1, linewidth=3)
plt.fill_between(range(len(p1)), downerror, y2=uperror, alpha=0.4)

# <headingcell level=3>

# Scikit-Learn : Machine Learning

# <codecell>

# http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html
# K-Means Clustering

# Let's load the Iris database
iris = datasets.load_iris()
X = iris.data # first two features only
y = iris.target # classes

# First five measurements, and all of the classes
print X[:5, :], "\n"
print y

kmeans = KMeans(n_clusters=len(np.unique(y)))
kmeans.fit(X, y)


fig = plt.figure()
ax = Axes3D(fig, elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, s=200)

fig = plt.figure()
ax = Axes3D(fig, elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=kmeans.predict(X), s=200)

# <codecell>

# http://scikit-learn.org/stable/tutorial/basic/tutorial.html#introduction
# SVC

# Let's load the digits database
digits = datasets.load_digits()

# What do these digits look like ?
for i in range(1, 6) :
    for j in range(3) :
        plt.subplot(3, 5, j*5 + i)
        plt.imshow(digits.images[j*3 + i], cmap="gray", interpolation="none")
        plt.grid(b=False)

# <codecell>

# I want to run some error analysis
# We're going to split the dataset into a variable number of train and test samples
# and from there, we'll see how the mean error changes with the train / test fraction

Success = [] # empty list
fraction = np.exp(np.linspace(np.log(0.01), np.log(0.8), 20)) # the fraction of the dataset to reserve for testing

for i in fraction :

    success = [] # empty list to keep our measurements for a given fraction
    
    for j in range(10) : # twenty random repetitions
        # Let's split the dataset randomly
        traindata, testdata, traintarget, testtarget = train_test_split(digits.data, digits.target, test_size = i)

        # Fit the classifier
        classifier = svm.SVC()
        classifier.fit(traindata, traintarget)

        # Calculate success rate
        success.append(np.sum(classifier.predict(testdata) == testtarget) / float(len(testtarget)))
    
    # Append all twenty measurements to our list 
    Success.append(success)
    print i # because I wanted to know how far we were...

# <codecell>

# OK, let's plot our results. First, let's calculate the mean success rate
meansuccess = np.mean(Success, axis=1)

# And bootstrap some confidence intervals
confint = [bootstrap.ci(np.array(Success)[i, :]) for i in range(len(meansuccess))]

# Drop our CIs around the plot
CIdown = [meansuccess[i] - confint[i][0] for i in range(len(meansuccess))]
CIup = [meansuccess[i] + confint[i][1] for i in range(len(meansuccess))]

# And plot
plt.plot(fraction, meansuccess)
plt.fill_between(fraction, CIdown, y2=CIup, alpha=0.4)

# <headingcell level=3>

# Scikit-Image : Image Processing

# <codecell>

# Let's open up an RGB histological image of a slice of liver
img = io.imread("liver.jpg")
io.imshow(img)
plt.grid(b=False)

# <codecell>

# We'll begin by sigmoid transforming the red channel, to enhance contrast
A = exposure.adjust_sigmoid(img[:, :, 0], cutoff=0.4, gain=30)
io.imshow(A)
plt.grid(b = False)

# <codecell>

# Let's threshold the image, based on the Otsu value
# We'll also invert the image in the process, to keep with image processing tradition :
# "Foreground" elements are 1, and background is 0
B = A < filter.threshold_otsu(A)
io.imshow(B)
plt.grid(b = False)

# <codecell>

# We have some noise. Let's remove small components :
C = morphology.remove_small_objects(B, 200)
io.imshow(C)
plt.grid(b = False)

# <codecell>

# Some of the nuclei are touching, and hence, segmenting based on objects won't give a good answer.
# We'll apply a smoothed distance transform first...
D = nd.distance_transform_edt(C)
io.imshow(D)
plt.grid(b = False)

# <codecell>

# ... and throw in a watershed algorithm to segment the objects
localmax = feature.peak_local_max(D, indices=False, min_distance=8, labels=C)
markers = nd.label(localmax)[0]
E = morphology.watershed(-D, markers, mask=C)
io.imshow(color.label2rgb(E, bg_label=0))
plt.grid(b = False)

# <codecell>

# Some basic stats ?
lab = np.unique(E) # list of different labels
props = measure.regionprops(E)
sizes = np.array([props[i].area for i in range(len(props))])
centroids = np.array([props[i].centroid for i in range(len(props))])
perimeter = np.array([props[i].perimeter for i in range(len(props))])

# Hacky cleanup of segmentation-induced noise
ind = sizes > 150
sizes = sizes[ind]
perimeter = perimeter[ind]
centroids = centroids[ind]

# Pairwise Internuclear Distribution
z = sp.distance.pdist(np.array(centroids))


plt.subplot(131)
plt.hist(sizes, 30)
plt.axvline(np.mean(sizes), c="r", linewidth=3)
plt.title("Size Distribution")

plt.subplot(132)
plt.hist(perimeter, 30)
plt.axvline(np.mean(perimeter), c="r", linewidth=3)
plt.title("Perimeter Distribution")

plt.subplot(133)
plt.hist(4 * np.pi * np.array(sizes) / np.array(perimeter)**2, 30)
plt.axvline(np.mean(4 * np.pi * np.array(sizes) / np.array(perimeter)**2), c="r", linewidth=3)
plt.title("Circularity Index")

plt.figure()
plt.hist(z, 30);
plt.title("Pairwise Internuclear Distance")

