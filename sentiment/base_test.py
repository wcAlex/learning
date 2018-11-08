import unittest
import submission

class TestSubmissionMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def test_extractWordFeatures(self):
        phi = submission.extractWordFeatures("I am what I am")
        self.assertEqual(2, phi["I"])
        self.assertEqual(2, phi["am"])
        self.assertEqual(1, phi["what"])
        self.assertEqual(3, len(phi))

if __name__ == '__main__':
    unittest.main()