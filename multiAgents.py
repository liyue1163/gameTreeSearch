# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
# Add for Q5
from game import Actions
from game import Grid

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"

        ghostPositions = successorGameState.getGhostPositions()
        food = currentGameState.getFood().asList()
        restFood = newFood.asList()

        deadDist = 999999
        for ghost in ghostPositions:
            if manhattanDistance(newPos, ghost) < deadDist:
                deadDist = manhattanDistance(newPos, ghost)

        min = 999999
        # A food is eaten
        if len(food) > len(restFood):
            if deadDist > 1:
                score = min
                return score
        if len(restFood) == 0:
            return 999999
        minFood = None
        min = 999999
        for food in restFood:
            if manhattanDistance(newPos, food) < min:
                minFood = food
                min = manhattanDistance(newPos, food)
        min = 999999 - min
        score = min
        if deadDist <= 1:
            score = -999999
        return score


        # return successorGameState.getScore()


# helper euclideanDistance
def euclideanDistance(position1, position2):
    "The euclidean distance heuristic for a PositionSearchProblem"

    return ((position1[0]-position2[0])**2+(position1[1]-position2[1])**2)**0.5

# helper manhattanDistance
def manhattanDistance(position1, position2):
    "The Manhattan distance heuristic for a PositionSearchProblem"

    return abs(position1[0] - position2[0]) + abs(position1[1] - position2[1])


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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def DFMinMax(self, curDepth, curState, curIndex):
        """
          A helper function
        """
        bestAct = None
        if curDepth == 0 or curState.isWin() or curState.isLose():
            return bestAct, self.evaluationFunction(curState)
        # Max player
        if curIndex == 0:
            value = -999999
        # Min player
        else:
            value = 999999
        actions = curState.getLegalActions(curIndex)
        for action in actions:
            nextState = curState.generateSuccessor(curIndex, action)
            nextIndex = (curIndex + 1) % curState.getNumAgents()
            nextDepth = curDepth
            if nextIndex == 0:
                nextDepth -= 1
            nextAct, score = self.DFMinMax(nextDepth, nextState, nextIndex)
            # Max player
            if curIndex == 0 and value < score:
                bestAct, value = action, score
            # Min player
            elif curIndex != 0 and value > score:
                bestAct, value = action, score
        return bestAct, value

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        act, value = self.DFMinMax(self.depth, gameState, self.index)

        return act

        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphaBeta(self, curDepth, curState, curIndex, alpha, beta):
        bestAct = None
        if curDepth == 0 or curState.isWin() or curState.isLose():
            return bestAct, self.evaluationFunction(curState)
        # Max player
        if curIndex == 0:
            value = -999999
        # Min player
        else:
            value = 999999
        actions = curState.getLegalActions(curIndex)
        for action in actions:
            nextState = curState.generateSuccessor(curIndex, action)
            nextIndex = (curIndex + 1) % curState.getNumAgents()
            nextDepth = curDepth
            if nextIndex == 0:
                nextDepth -= 1
            nextAct, score = self.alphaBeta(nextDepth, nextState, nextIndex, alpha, beta)
            # Max player
            if curIndex == 0:
                if  value < score:
                    bestAct, value = action, score
                if value >= beta:
                    return bestAct, value
                alpha = max(alpha, value)
            # Min player
            elif curIndex != 0:
                if value > score:
                    bestAct, value = action, score
                if value <= alpha:
                    return bestAct, value
                beta = min(beta, value)
        return bestAct, value

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -999999
        beta = 999999

        act, value = self.alphaBeta(self.depth, gameState, self.index, alpha, beta)

        return act
        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectiMax(self, curDepth, curState, curIndex):
        bestAct = None
        # if curState.isWin():
            # print("Win!")
        if curDepth == 0 or curState.isWin() or curState.isLose():
            return bestAct, self.evaluationFunction(curState)
        # Max player
        if curIndex == 0:
            value = -999999
        # Exp player
        else:
            value = 0
        actions = curState.getLegalActions(curIndex)
        # if curIndex == 0:
            # print(actions)
        for action in actions:
            # if curIndex == 0:
                # print(action)
            nextState = curState.generateSuccessor(curIndex, action)
            nextIndex = (curIndex + 1) % curState.getNumAgents()
            nextDepth = curDepth
            if nextIndex == 0:
                nextDepth -= 1
            nextAct, score = self.expectiMax(nextDepth, nextState, nextIndex)
            #if nextAct != None:
                #print(nextAct + ": " + str(score))
            # Exp player
            if curIndex == 0 and value < score:
                bestAct, value = action, score

            elif curIndex != 0:
                value += float(1)/float(len(actions)) * score
        #if curIndex == 0:
            #print("bestAct: " + bestAct + " bestScore: " + str(score))
        return bestAct, value

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        act, value = self.expectiMax(self.depth, gameState, self.index)

        return act
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    currentScore = scoreEvaluationFunction(currentGameState)

    if currentGameState.isLose():
        return -float("inf")
    elif currentGameState.isWin():
        # print("Win!")
        return currentScore

    # food distance
    foodlist = currentGameState.getFood().asList()
    distanceToClosestFood = 0
    minDist = 999999
    nearestFood = None
    for food in foodlist:
        if manhattanDistance(pos, food) < minDist:
            minDist = manhattanDistance(pos, food)
            nearestFood = food
    foodProb = PositionSearchProblem(currentGameState, start=pos, goal=nearestFood, warn=False, visualize=False)
    distanceToClosestFood = len(breadthFirstSearch(foodProb))

    numberOfCapsulesLeft = len(currentGameState.getCapsules())

    # number of foods left
    numberOfFoodsLeft = len(foodlist)

    # active ghosts, scared ghosts and their positions
    numGhosts = currentGameState.getNumAgents() - 1
    scaredGhosts, activeGhosts = {}, {}
    for ghost in range(1, numGhosts+1):
        state = currentGameState.getGhostState(ghost)
        if state.scaredTimer:
            scaredGhosts[ghost] = currentGameState.getGhostPosition(ghost)
        else:
            activeGhosts[ghost] = currentGameState.getGhostPosition(ghost)

    # min distance to active ghost
    if len(activeGhosts) == 0:
        distanceToClosestActiveGhost = float("inf")
    else:
        dist = []
        for ghost in activeGhosts:
            ghostProb = PositionSearchProblem(currentGameState, start=pos, goal=activeGhosts[ghost], warn=False, visualize=False)
            dist.append(len(breadthFirstSearch(ghostProb)))
        distanceToClosestActiveGhost = min(dist)
    if distanceToClosestActiveGhost == 0:
        distanceToClosestActiveGhost = 0.5

    # min distance to scared ghost
    if len(scaredGhosts) == 0:
        distanceToClosestScaredGhost = 0
    else:
        dist = []
        for ghost in scaredGhosts:
            ghostProb = PositionSearchProblem(currentGameState, start=pos, goal=scaredGhosts[ghost], warn=False, visualize=False)
            dist.append(len(breadthFirstSearch(ghostProb)))
        distanceToClosestScaredGhost = min(dist)

    score =   1 * currentScore + \
              -0.25 * distanceToClosestFood + \
              -2 * (1./distanceToClosestActiveGhost) + \
              -2 * distanceToClosestScaredGhost + \
              -25 * numberOfCapsulesLeft + \
              -5 * numberOfFoodsLeft
    return score
    # util.raiseNotDefined()


# Helper class
class PositionSearchProblem:
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1]) <= 1

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    seen = {problem.getStartState():0}
    OPEN = util.Queue()
    # path is a list of (state, action, cost)
    path = [(problem.getStartState(), None, None)]
    OPEN.push(path)
    while not OPEN.isEmpty():
        path = OPEN.pop()
        state = path[-1][0]
        if problem.isGoalState(state):
            actions = []
            for i in range(1, len(path)):
                actions.append(path[i][1])
            return actions
        successors = problem.getSuccessors(state)
        for successor in successors:
            nextPath = path[:]
            nextPath.append(successor)
            actions = []
            # cycle check
            newState = successor[0]
            for i in range(1, len(nextPath)):
                actions.append(nextPath[i][1])
            cost = problem.getCostOfActions(actions)
            if not newState in seen or cost < seen[newState]:
                seen[newState] = cost
                OPEN.push(nextPath)
    print("Should not reach here")
# Abbreviation
better = betterEvaluationFunction
