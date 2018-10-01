import unittest
import submission
from collections import defaultdict
from sets import Set

class TestStringMethods(unittest.TestCase):

    def test_findAlphabeticallyLastWord(self):
        word = submission.findAlphabeticallyLastWord("abc bcd def hig")
        self.assertEqual(word, 'hig')

    def test_euclideanDistance(self):
        self.assertEqual(5, submission.euclideanDistance((3,3), (6, 7)))

    def test_mutateSentences(self):
        sentense = 'the cat and the mouse'
        res = submission.mutateSentences(sentense)
        self.assertEqual(4, len(res))
        self.assertTrue('and the cat and the' in res)
        self.assertTrue('the cat and the mouse' in res)
        self.assertTrue('the cat and the cat' in res)
        self.assertTrue('cat and the cat and' in res)

    def test_sparseVectorDotProduct(self):
        v1 = defaultdict(float)
        v2 = defaultdict(float)
        v1["x"] = 1
        v1["y"] = 3
        v1["z"] = -5
        v2["x"] = 4
        v2["y"] = -2
        v2["z"] = -1

        res = submission.sparseVectorDotProduct(v1, v2)
        self.assertEqual(3, res)

    def test_incrementSparseVector(self):
        v1 = defaultdict(float)
        v2 = defaultdict(float)
        v1["x"] = 1
        v1["y"] = 3
        v1["z"] = -5
        v2["x"] = 4
        v2["y"] = -2
        v2["z"] = -1

        submission.incrementSparseVector(v1, 2, v2)
        self.assertEqual(9, v1["x"])
        self.assertEqual(-1, v1["y"])
        self.assertEqual(-7, v1["z"])

    def test_findSingletonWords(self):
        text = "hello world, this is a new day, hello"
        res = submission.findSingletonWords(text)
        self.assertEqual(6, len(res))

    def test_computeLongestPalindromeLength(self):
        text = "animal"
        ans = submission.computeLongestPalindromeLength(text)

        self.assertEqual(3, ans) # "ama" for animal

if __name__ == '__main__':
    unittest.main()