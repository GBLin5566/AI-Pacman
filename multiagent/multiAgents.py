# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

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
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    currentScore = successorGameState.getScore()
    minGhostDis = minGhost(newPos, newGhostStates)
    food = greedyFood(newPos, oldFood)
    avgFoodDis =avgFood(newPos, oldFood)
    mazeSize = (len(list(oldFood)), len(list(oldFood[1])))

    toReturn = calculateFunction([currentScore, minGhostDis, minGhostDis/float(mazeSize[0]*mazeSize[1]),food, avgFoodDis])
    return toReturn

def calculateFunction(calc):
    if calc[1] <= 1:
        return float('-inf')
    return calc[0] + calc[3] + 11.0/calc[4]

def minGhost(pos, ghostStates):
    distances = float('inf')
    for ghostState in ghostStates:
        ghostCoor = ghostState.getPosition()
        distances = min(distances, manhattanDistance(pos, ghostCoor))
    return max(1, distances)


def greedyFood(newPos, oldFood):
    count = 0
    frame_size = 2
    for x in xrange(newPos[0]-frame_size, newPos[0] + frame_size + 1):
        # Do not point out of the array
        for y in xrange(newPos[1]-frame_size, newPos[1] + frame_size + 1):
            if (0 <= x and x < len (list(oldFood))) and (0<= y and y < len(list(oldFood[1]))) and oldFood[x][y] :
                count += 1
    return count


def avgFood(newPos, oldFood):
    distances = []
    for x, row in enumerate(oldFood):
        for y, col in enumerate(oldFood[x]):
            if oldFood[x][y]:
                distances.append(manhattanDistance(newPos, (x, y)))
    return max(1, sum(distances)/ float(len(distances)))


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

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

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
    """
    "*** YOUR CODE HERE ***"
    totalAgents = gameState.getNumAgents() - 1
    pacmanIndex = 0
    startDepth = 1
    totalDepth = self.depth
    def MiniMax(gameState, depth, index):
        if depth > totalDepth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalMoves = [action for action in gameState.getLegalActions(index) if action != Directions.STOP]
        nextIndex, nextDepth = (index+1, depth) if index+1 <= totalAgents else (0, depth + 1)
        outputs = [MiniMax(gameState.generateSuccessor(index, action), nextDepth, nextIndex) for action in legalMoves]
        if depth == 1 and index == 0:
            best = max(outputs)
            bestIndices = [index for index in xrange(len(outputs)) if outputs[index] == best]
            chosenIndex = random.choice(bestIndices)
            return legalMoves[chosenIndex]
        return max(outputs) if index==0 else min(outputs)

    return MiniMax(gameState, startDepth, pacmanIndex)


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    MAX_INT = float('inf')
    MIN_INT = float('-inf')
    totalAgents = gameState.getNumAgents() - 1
    pacmanIndex = 0
    startDepth = 1
    totalDepth = self.depth
    def AlphaBeta(gameState, depth, index, alpha, beta):
        if depth > totalDepth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalMoves = [action for action in gameState.getLegalActions(index) if action != Directions.STOP]
        nextIndex, nextDepth = (index+1, depth) if index+1 <= totalAgents else (0, depth + 1)
        if depth == 1 and index == 0:
            outputs = [AlphaBeta(gameState.generateSuccessor(index, action), nextDepth, nextIndex, alpha, beta) for action in legalMoves]
            best = max(outputs)
            bestIndices = [index for index in xrange(len(outputs)) if outputs[index] == best]
            chosenIndex = random.choice(bestIndices)
            return legalMoves[chosenIndex]
        if index == 0:
            best = MIN_INT
            for action in legalMoves:
                best = max(best, AlphaBeta(gameState.generateSuccessor(index, action), nextDepth, nextIndex, alpha, beta))
                if best >= beta: # Over the window
                    return best
                alpha = max(alpha, best)
            return best
        else:
            best = MAX_INT
            for action in legalMoves:
                best = min(best, AlphaBeta(gameState.generateSuccessor(index, action), nextDepth, nextIndex, alpha, beta))
                if best <= alpha: # Over the window
                    return best
                beta = min(beta, best)
            return best
    return AlphaBeta(gameState, startDepth, pacmanIndex, MIN_INT, MAX_INT)

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    totalAgents = gameState.getNumAgents() - 1
    pacmanIndex = 0
    startDepth = 1
    totalDepth = self.depth
    def ExpectiMax(gameState, depth, index):
        if depth > totalDepth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalMoves = [action for action in gameState.getLegalActions(index) if action != Directions.STOP]
        nextIndex, nextDepth = (index+1, depth) if index+1 <= totalAgents else (0, depth + 1)
        outputs = [ExpectiMax(gameState.generateSuccessor(index, action), nextDepth, nextIndex) for action in legalMoves]
        if depth == 1 and index == 0:
            best = max(outputs)
            bestIndices = [index for index in xrange(len(outputs)) if outputs[index] == best]
            chosenIndex = random.choice(bestIndices)
            return legalMoves[chosenIndex]
        return max(outputs) if index==0 else sum(outputs)/float(len(outputs))

    return ExpectiMax(gameState, startDepth, pacmanIndex)

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  if currentGameState.isWin() or currentGameState.isLose():
      return float('inf') if currentGameState.isWin() else float('-inf')
  # Maze Info.
  gameState = currentGameState
  statePosition = gameState.getPacmanPosition()
  stateFood = gameState.getFood()
  stateFoodSum = gameState.getNumFood()
  stateGhost = gameState.getGhostStates()
  stateWall = gameState.getWalls()
  stateScore = gameState.getScore()
  stateCapsule = gameState.getCapsules()

  '''
    Main concept: Input the key information, and tune the weight of these information
  '''

  #print stateFood.asList()
  '''Food'''
  foodList = stateFood.asList()
  minFood = min(manhattanDistance(statePosition, food) for food in foodList)
  '''Ghost'''
  ghostLocation = [ghost.getPosition() for ghost in stateGhost]
  minGhostDistance = manhattanDistance(min(ghostLocation), statePosition)
  scaredTime = [ghost.scaredTimer for ghost in stateGhost]
  whiteGhost = [ghost for ghost in stateGhost if ghost.scaredTimer != 0]
  normalGhost = [ghost for ghost in stateGhost if ghost.scaredTimer == 0]
  centerGhost = (sum(ghost[0] for ghost in ghostLocation) / len(ghostLocation), sum(ghost[1] for ghost in ghostLocation) / len(ghostLocation))
  centerGhostDistance = manhattanDistance(centerGhost, statePosition)

  '''Capsule'''
  capsuleDistance = [manhattanDistance(statePosition, cap) for cap in stateCapsule]
  minCapsuleDistance = min(capsuleDistance) if len(capsuleDistance) > 0 else float('inf')

  '''Score'''
  foodScore, ghostScore, capsuleScore, hunterScore = 0,0,0,0
  foodW, ghostW, capsuleW, hunterW = 1.0,1.0,1.0,1.0

  '''Factor'''
  Factor_minGhost = 1
  Factor_hunt_active_distance = 6
  Factor_minCapsule = 3
  Factor_dead = 0
  if len(whiteGhost) == 0:
      if minGhostDistance >= Factor_minGhost and minGhostDistance <= Factor_hunt_active_distance and centerGhostDistance <= minGhostDistance:
          ghostW, ghostScore = 10.0, (-1.0 / max(1, centerGhostDistance))
          if len(capsuleDistance) > 0 and minCapsuleDistance <= Factor_minCapsule:
              capsuleW, capsuleScore = 20.0, (1.0 / max(1, minCapsuleDistance))
              ghostW, ghostScore = 1.5, (-1.0 / max(1, centerGhostDistance))
      elif centerGhostDistance >= minGhostDistance and minGhostDistance >= 1:
          foodW = 5.0
          ghostW, ghostScore = 5.0, (-1.0 / max(1, minGhostDistance))
          if len(capsuleDistance) > 0 and minCapsuleDistance <= Factor_minCapsule:
              capsuleW, capsuleScore = 20.0, (1.0 / max(1, minCapsuleDistance))
              ghostW, ghostScore = 1.5, (-1.0 / max(1, minGhostDistance))
      elif minGhostDistance == Factor_minGhost:
          ghostW, ghostScore = 25.0 , (-1.0/ max(1, minGhostDistance))
      elif minGhostDistance == Factor_dead:
          return float('-inf')
      else:
          ghostW, ghostScore = 5.0, (-1.0/max(1, minGhostDistance))
  else:
      minPrey = min(manhattanDistance(ghost.getPosition(), statePosition) for ghost in whiteGhost)
      if len(normalGhost) > 0:
          minGhostDistance = min(manhattanDistance(ghost.getPosition(), statePosition) for ghost in normalGhost)
          centerGhost = sum(ghost.getPosition()[0] for ghost in normalGhost) / len(normalGhost), sum(ghost.getPosition()[1] for ghost in normalGhost) / len(normalGhost)
          centerGhostDistance = manhattanDistance(centerGhost, statePosition)
          if centerGhostDistance <= minGhostDistance and minGhostDistance >= Factor_minGhost and minGhostDistance <= Factor_hunt_active_distance:
              ghostW, ghostScore = 10.0, (-1.0/max(centerGhostDistance, 1))
          elif centerGhostDistance >= minGhostDistance and minGhostDistance >= 1:
              ghostW, ghostScore = 5.0 , (-1.0/ max(1, centerGhostDistance))
          elif minGhostDistance == Factor_dead:
              return float('-inf')
          elif minGhostDistance == Factor_minGhost:
              ghostW, ghostScore = 15.0 , (-1.0/max(1, minGhostDistance))
          else:
              ghostScore = -1.0/max(1, minGhostDistance)
      hunterW, hunterScore = 35.0 , (1.0 / max(1, minPrey))

  func = stateScore + foodW * foodScore + ghostW * ghostScore + capsuleW * capsuleScore + hunterW * hunterScore
  return func


# Abbreviation
better = betterEvaluationFunction


