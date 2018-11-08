import shell
import util
import wordsegUtil

vowels = set(['a', 'e', 'i', 'o', 'u'])

############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        return 0, 0 # word [start_pos, end_pos), len = end_pos - start_pos

        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        if state[1] == len(self.query):
            return True
        else:
            return False
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        results = []

        for endPos in range(state[1]+1, len(self.query)+1):
            word = self.query[state[1] : endPos]
            cost = self.unigramCost(word)
            results.append((word, (state[1], endPos), cost)) # action, newState, cost

        return results
        # END_YOUR_CODE

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 10 lines of code, but don't worry if you deviate from this)
    return ' '.join(ucs.actions)
    # END_YOUR_CODE

############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (wordsegUtil.SENTENCE_BEGIN, -1) # (cur_word, cur_index)
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        if state[1] == len(self.queryWords) - 1:
            return True
        else:
            return False
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 16 lines of code, but don't worry if you deviate from this)
        curWord = state[0]
        nextWordIndex = state[1]+1
        nextRawWord = self.queryWords[nextWordIndex]

        nextWords = self.possibleFills(nextRawWord)
        if len(nextWords) == 0:
            nextWords.add(nextRawWord)

        results = []
        for nextWord in nextWords:
            cost = self.bigramCost(curWord, nextWord)
            results.append((nextWord, (nextWord, nextWordIndex), cost)) # action, newState, cost

        return results
        # END_YOUR_CODE

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    if len(queryWords) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))

    return ' '.join(ucs.actions)
    # END_YOUR_CODE

############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        return (wordsegUtil.SENTENCE_BEGIN, 0)  # (vowel_word, end_pos) [start_pos, end_pos) in char sequence, word len = end_pos - start_pos
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        if state[1] == len(self.query):
            return True
        else:
            return False
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 23 lines of code, but don't worry if you deviate from this)
        results = []

        for endPos in range(state[1] + 1, len(self.query) + 1):
            beginPos = state[1]
            cword = self.query[beginPos: endPos]

            nextWords = self.possibleFills(cword)
            if len(nextWords) == 0:
                continue

            for nextWord in nextWords:
                if not self.isValidateWord(nextWord):
                    continue

                cost = self.bigramCost(state[0], nextWord)
                results.append((nextWord, (nextWord, endPos), cost))  # action, newState, cost

        return results

        # END_YOUR_CODE

    def isValidateWord(self, word):
        for c in word:
            if c in vowels:
                return True

        return False

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 11 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))

    res = ' '.join(ucs.actions)
    return res
    # END_YOUR_CODE

############################################################

if __name__ == '__main__':
    shell.main()
