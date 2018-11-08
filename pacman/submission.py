from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from game import Actions


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
        ------------------------------------------------------------------------------
        Description of GameState and helper functions:

        A GameState specifies the full game state, including the food, capsules,
        agent configurations and score changes. In this function, the |gameState| argument
        is an object of GameState class. Following are a few of the helper methods that you
        can use to query a GameState object to gather information about the present state
        of Pac-Man, the ghosts and the maze.

        gameState.getLegalActions():
            Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

        gameState.generateSuccessor(agentIndex, action):
            Returns the successor state after the specified agent takes the action.
            Pac-Man is always agent 0.

        gameState.getPacmanState():
            Returns an AgentState object for pacman (in game.py)
            state.configuration.pos gives the current position
            state.direction gives the travel vector

        gameState.getGhostStates():
            Returns list of AgentState objects for the ghosts

        gameState.getNumAgents():
            Returns the total number of agents in the game

        gameState.getScore():
            Returns the score corresponding to the current state of the game


        The GameState class is defined in pacman.py and you might want to look into that for
        other helper methods, though you don't need to.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (oldFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (problem 1)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction. Terminal states can be found by one of the following:
          pacman won, pacman lost or there are no legal moves.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.getScore():
            Returns the score corresponding to the current state of the game

          gameState.isWin():
            Returns True if it's a winning state

          gameState.isLose():
            Returns True if it's a losing state

          self.depth:
            The depth to which search should continue

        """
        # minmax algorithm.
        # s is current state.
        # d is search depth.
        # p is player index.
        def minmax(s, d, p):
            # End State (Win or Lose)
            if s.isWin() or s.isLose():
                return [s.getScore(), Directions.STOP]

            if d == 0:
                return [self.evaluationFunction(s), None]

            # Pacman is the maximizing player
            if p == 0:
                maxAction = [float("-inf"), Directions.STOP]
                actions = s.getLegalActions(p)
                for action in actions:
                    newVal = minmax(s.generateSuccessor(p, action), d, 1)[0]
                    if newVal > maxAction[0]:
                        maxAction[0] = newVal
                        maxAction[1] = action
                return maxAction
            else:
                minAction = [float("inf"), Directions.STOP]
                newVal = float("inf")
                nextAgentIdx = p + 1
                actions = s.getLegalActions(p)
                for action in actions:
                    # check if next agent is pacman.
                    if nextAgentIdx == s.getNumAgents():
                        newVal = minmax(s.generateSuccessor(p, action), d - 1, 0)[0]
                    else:
                        newVal = minmax(s.generateSuccessor(p, action), d, nextAgentIdx)[0]

                    if newVal < minAction[0]:
                        minAction[0] = newVal
                        minAction[1] = action

                return minAction

        res = minmax(gameState, self.depth, 0)
        return res[1]

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (problem 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        # BEGIN_YOUR_CODE (our solution is 49 lines of code, but don't worry if you deviate from this)

        # minmax AlphaBeta prune algorithm.
        # s is current state.
        # d is search depth.
        # p is player index.
        # alpha is the best max value in the path.
        # beta is the best min value in the path.
        def minmaxAlphaBeta(s, d, p, alpha, beta):
            # End State (Win or Lose)
            if s.isWin() or s.isLose():
                return [s.getScore(), Directions.STOP]

            if d == 0:
                return [self.evaluationFunction(s), None]

            # Pacman is the maximizing player
            if p == 0:
                maxAction = [float("-inf"), Directions.STOP]
                actions = s.getLegalActions(p)
                for action in actions:
                    newVal = minmaxAlphaBeta(s.generateSuccessor(p, action), d, 1, alpha, beta)[0]
                    if newVal > maxAction[0]:
                        maxAction[0] = newVal
                        maxAction[1] = action

                        alpha = max(alpha, newVal)
                        if alpha >= beta:
                            break

                return maxAction
            else:
                minAction = [float("inf"), Directions.STOP]
                newVal = float("inf")
                nextAgentIdx = p + 1
                actions = s.getLegalActions(p)
                for action in actions:
                    # check if next agent is pacman.
                    if nextAgentIdx == s.getNumAgents():
                        newVal = minmaxAlphaBeta(s.generateSuccessor(p, action), d - 1, 0, alpha, beta)[0]
                    else:
                        newVal = minmaxAlphaBeta(s.generateSuccessor(p, action), d, nextAgentIdx, alpha, beta)[0]

                    if newVal < minAction[0]:
                        minAction[0] = newVal
                        minAction[1] = action

                        beta = min(beta, newVal)
                        if alpha >= beta:
                            break

                return minAction

        res = minmaxAlphaBeta(gameState, self.depth, 0, float("-inf"), float("inf"))
        return res[1]
        # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (problem 3)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
        def exptmax(s, d, p):
            # End State (Win or Lose)
            if s.isWin() or s.isLose():
                return [s.getScore(), Directions.STOP]

            if d == 0:
                return [self.evaluationFunction(s), None]

            # Pacman is the maximizing player
            if p == 0:
                maxAction = [float("-inf"), Directions.STOP]
                actions = s.getLegalActions(p)
                for action in actions:
                    newVal = exptmax(s.generateSuccessor(p, action), d, 1)[0]
                    if newVal > maxAction[0]:
                        maxAction[0] = newVal
                        maxAction[1] = action
                return maxAction
            else:
                newVal = 0.0
                nextAgentIdx = p + 1
                actions = s.getLegalActions(p)
                prob = 1.0 / len(actions)
                for action in actions:
                    # check if next agent is pacman.
                    if nextAgentIdx == s.getNumAgents():
                        newVal += prob * exptmax(s.generateSuccessor(p, action), d - 1, 0)[0]
                    else:
                        newVal += prob * exptmax(s.generateSuccessor(p, action), d, nextAgentIdx)[0]

                return [newVal, Directions.STOP]

        res = exptmax(gameState, self.depth, 0)
        return res[1]

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
    """
      Your extreme, unstoppable evaluation function (problem 4).

      DESCRIPTION: <write something here so we know what you did>

      Score of a state is measured by below features:
      1. Closest ghost agent distance from pacman.
      2. Closest capsule distance from pacman.
      3. Closest food distance from pacman.
      4. Total (sum) food distance from pacman.

      Score strategy:
      1. get the base score from game score.
      2. update score by things surround pacman to gain local optimal. (closest food, ghost, capsule)
      3. increase score to encourage pacman to eat nearby scared ghost.
      4. decrease score to urge pacman keep away from active ghost.
      4. decrease scroe to urge pacman to move towards food.

    """

    # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)

    # find the closest food distance.
    def closestFoodDistance(pacmanPos):
        foodDists = []
        for foodPos in currentGameState.getFood().asList(key=True):
            foodDists.append(util.manhattanDistance(pacmanPos, foodPos))
        return min(foodDists) if len(foodDists) > 0 else 1

    # calculate total food distance.
    def foodTotalDistance(pacmanPos):
        foodDist = []
        for foodPos in currentGameState.getFood().asList(key=True):
            foodDist.append(util.manhattanDistance(pacmanPos, foodPos))
        return sum(foodDist)

    # get closest ghost distance.
    def closestActiveGhostDistance(pacmanPos):
        ghostDists = []
        for ghostState in currentGameState.getGhostStates():
            ghostDists.append(util.manhattanDistance(pacmanPos, ghostState.getPosition()))
        return min(ghostDists) if len(ghostDists) > 0 else 999999

    # get closet capsule distance.
    def closestCapsuleDistance(pacmanPos):
        capsuleDist = []
        for capsPos in currentGameState.getCapsules():
            capsuleDist.append(util.manhattanDistance(pacmanPos, capsPos))
        return min(capsuleDist) if len(capsuleDist) > 0 else 999999

    # get base score according to pacman's current position and things around it.
    def scorePacmanWithNearbySituation():
        pacmanPos = currentGameState.getPacmanPosition()
        score = currentGameState.getScore()
        if closestCapsuleDistance(pacmanPos) < closestActiveGhostDistance(pacmanPos):
            return score + 40
        if closestFoodDistance(pacmanPos) < closestActiveGhostDistance(pacmanPos) + 3:
            return score + 20
        if closestCapsuleDistance(pacmanPos) < closestFoodDistance(pacmanPos) + 3:
            return score + 30
        else:
            return score

    # reward pacman if nearby ghost is scared.
    def getGhostScaredReward():
        pacmanPos = currentGameState.getPacmanPosition()
        rewards = []
        for ghostState in currentGameState.getGhostStates():
            if ghostState.scaredTimer > 7 and util.manhattanDistance(ghostState.getPosition(), pacmanPos) <= 4:
                rewards.append(50)
            if ghostState.scaredTimer > 7 and util.manhattanDistance(ghostState.getPosition(), pacmanPos) <= 3:
                rewards.append(60)
            if ghostState.scaredTimer > 7 and util.manhattanDistance(ghostState.getPosition(), pacmanPos) <= 2:
                rewards.append(70)
            if ghostState.scaredTimer > 7 and util.manhattanDistance(ghostState.getPosition(), pacmanPos) <= 1:
                rewards.append(90)

        return max(rewards) if len(rewards) > 0 else 0

    # toll the state by active ghosts.
    def getGhostRiskToll():
        pacmanPos = currentGameState.getPacmanPosition()
        toll = []
        for ghostState in currentGameState.getGhostStates():
            if ghostState.scaredTimer == 0:
                toll.append(-1 * util.manhattanDistance(pacmanPos, ghostState.getPosition()) - 20)

        return min(toll) if len(toll) > 0 else 0

    pacmanPos = currentGameState.getPacmanPosition()

    score = scorePacmanWithNearbySituation()
    score += getGhostScaredReward()
    score -= getGhostRiskToll()
    score -= .4 * foodTotalDistance(pacmanPos)

    return score

# Abbreviation
better = betterEvaluationFunction
