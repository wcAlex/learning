import collections
import math
from sets import Set
from collections import defaultdict

############################################################
# Problem 3a

def findAlphabeticallyLastWord(text):
    """
    Given a string |text|, return the word in |text| that comes last
    alphabetically (that is, the word that would appear last in a dictionary).
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find max() and list comprehensions handy here.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)

    words = text.split()
    candidate = ""
    for i, word in enumerate(words):
        if i == 0:
            candidate = word
            continue

        if max(candidate, word) != candidate:
            candidate = word

    return candidate
    # END_YOUR_CODE

############################################################
# Problem 3b

def euclideanDistance(loc1, loc2):
    """
    Return the Euclidean distance between two locations, where the locations
    are pairs of numbers (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

    # END_YOUR_CODE

############################################################
# Problem 3c

def mutateSentences(sentence):
    """
    Given a sentence (sequence of words), return a list of all "similar"
    sentences.
    We define a sentence to be similar to the original sentence if
      - it as the same number of words, and
      - each pair of adjacent words in the new sentence also occurs in the original sentence
        (the words within each pair should appear in the same order in the output sentence
         as they did in the orignal sentence.)
    Notes:
      - The order of the sentences you output doesn't matter.
      - You must not output duplicates.
      - Your generated sentence can use a word in the original sentence more than
        once.
    Example:
      - Input: 'the cat and the mouse'
      - Output: ['and the cat and the', 'the cat and the mouse', 'the cat and the cat', 'cat and the cat and']
                (reordered versions of this list are allowed)
    """
    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    words = sentence.split()
    cache = {} # string => set

    def buildSentences(candidate):
        res = Set()
        curPath = candidate.split()

        if candidate in cache:
            return cache[candidate]

        if len(curPath) == len(sentence.split()):
            res.add(candidate)
            return res

        for word in words:
            if len(candidate) == 0:
                res = res.union(buildSentences(word))
            elif curPath[-1] + " " + word in sentence:
                res = res.union(buildSentences(candidate + " " + word))

        cache[candidate] = res
        return res

    ans = buildSentences("")
    return list(ans)

    # END_YOUR_CODE

############################################################
# Problem 3d

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collections.defaultdict(float), return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    pairs = zip(v1.values(), v2.values())
    return sum( i*j for i,j in pairs)

    # END_YOUR_CODE

############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    for key in v1.keys():
        v1[key] = v1[key] + scale * v2[key]

    for key in v2.keys():
        if key not in v1:
            v1[key] = scale * v2[key]
    # END_YOUR_CODE

############################################################
# Problem 3f

def findSingletonWords(text):
    """
    Splits the string |text| by whitespace and returns the set of words that
    occur exactly once.
    You might find it useful to use collections.defaultdict(int).
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    d = defaultdict(int)
    for w in text.split():
        d[w] += 1

    return [k for k in d.keys() if d[k] == 1]
    # END_YOUR_CODE

############################################################
# Problem 3g

def computeLongestPalindromeLength(text):
    """
    A palindrome is a string that is equal to its reverse (e.g., 'ana').
    Compute the length of the longest palindrome that can be obtained by deleting
    letters from |text|.
    For example: the longest palindrome in 'animal' is 'ama'.
    Your algorithm should run in O(len(text)^2) time.
    You should first define a recurrence before you start coding.
    """
    # BEGIN_YOUR_CODE (our solution is 19 lines of code, but don't worry if you deviate from this)

    cache = {} # (i,j) => i
    def countLongestPalindrome(s, e, str):
        if (s, e) in cache:
            return cache[(s, e)]

        if s > e:
            return 0
        if s == e:
            return 1

        res = 0
        if str[s] == str[e]:
            res = 2 + countLongestPalindrome(s+1, e-1, str)
        else:
            res = max(countLongestPalindrome(s+1, e, str), countLongestPalindrome(s, e-1, str))
        cache[(s, e)] = res

        return res

    return countLongestPalindrome(0, len(text)-1, text)
    # END_YOUR_CODE
