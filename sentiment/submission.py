#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)

    phi = collections.defaultdict(int)
    tokens = x.split()
    for word in tokens:
        phi[word] += 1

    return phi
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''

    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    def sdF(x, w, y):
        if dotProduct(x, w)*y < 1:
            return {k : -1 * v * y for k, v in x.items()}
        else:
            return {k : 0 for k, v in x.items()}

    def predict(x):
        return 1 if dotProduct(featureExtractor(x), weights) >= 0 else -1

    for t in range(numIters):
        for s in trainExamples:
            features = featureExtractor(s[0])
            gradient = sdF(features, weights, s[1])
            increment(weights, -1 * eta, gradient)

        er = evaluatePredictor(trainExamples, predict)
        print 'train example error rate %f' % (er)
        er = evaluatePredictor(testExamples, predict)
        print 'test example error rate %f' % (er)

    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = {}

        for k, v in random.sample(weights.items(), random.randint(1, len(weights))):
            phi[k] = random.randint(1, 5)

        y= 1 if dotProduct(phi, weights) >= 0 else 0

        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        d = dict()
        str = x.strip(' \t\n\r').replace(" ", "")
        for i in range(len(x)):
            if i + n >= len(x):
                break
            w = str[i:i+n]
            if w in d:
                d[w] += 1
            else:
                d[w] = 1

        return d

        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)

    def shouldStop(oldCentroids, centroids, currentIterations, maxIterations):
        if currentIterations > maxIterations: return True
        return oldCentroids == centroids

    # Return list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j).
    def getAssignment(dataSet, centroids, cache):
        newassign = [0]*len(dataSet)
        loss = 0.0
        for i, datapoint in enumerate(dataSet):
            k = 0
            mindist = sys.float_info.max
            for idx, centroid in enumerate(centroids):
                dist = distance(centroid, datapoint, cache[idx])
                if dist < mindist:
                    k = idx
                    mindist = dist
            newassign[i] = k
            loss += mindist

        return newassign, loss

    # calculate distance between centroid and datapoint.
    def distance(centroid, datapoint, centroidSize):
        dist = 0.0
        for k, v in datapoint.items():
            if v != 0:
                dist = (centroid.get(k, 0) - v)**2 - centroid.get(k, 0)**2
        return dist + centroidSize

    # Return new centroids.
    # assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j).
    def getCentroids(assignments, dataSet, clusters, cache):

        centroids = [dict() for x in range(clusters)]  #list of centroids
        count = [0]*clusters # count per centroid
        for i, c in enumerate(assignments): #i: index of data point, c: index of cluster
            count[c] += 1
            for k, v in dataSet[i].items(): #k: dimension, v:value
                centroids[c][k] = centroids[c].get(k, 0) + v

        for i, centroid in enumerate(centroids):
            dist = 0.0
            for k in centroid.keys():
                centroid[k] /= 1.0 * count[i]
                dist += centroid[k]**2
            cache[i] = dist

        return centroids

    # Initial setup
    newCentroids = random.sample(examples, K)
    cache = {}
    for i, c in enumerate(newCentroids):
        cache[i] = 0.0
        for k, v in c.items():
            cache[i] += v ** 2

    iteration = 0
    oldCentroids = None
    loss = 0.0
    assignments = []

    while not shouldStop(oldCentroids, newCentroids, iteration, maxIters):
        assignments, loss = getAssignment(examples, newCentroids, cache)
        oldCentroids = newCentroids
        newCentroids = getCentroids(assignments, examples, K, cache)
        iteration += 1

    # TODO: Add Loss/Error Rate.
    return (newCentroids, assignments, loss)
    # END_YOUR_CODE
