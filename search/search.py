# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

  "My Own Lib."
def checkGoal(problem, state):
  return problem.isGoalState(state)

def getSuccessors(problem, state):
  return problem.getSuccessors(state)

def returnDirection(stack):
    from game import Directions
    s = Directions.SOUTH
    n = Directions.NORTH
    w = Directions.WEST
    e = Directions.EAST
    decisionList = []
    tmp = 0
    for a in stack.list:
        if tmp == 0:
            tmp = a
            continue
        elif tmp[0] - a[0] == 1:
            decisionList.append(w)
        elif tmp[0] - a[0] == -1:
            decisionList.append(e)
        elif tmp[1] - a[1] == 1:
            decisionList.append(s)
        else:
            decisionList.append(n)
        tmp = a
    return decisionList

    "Try to use recursive is such a stupid thing"
def DFS(problem, stack, state):
  if checkGoal(problem, state):
      stack.push(state)
      return "Goal"
  succ = getSuccessors(problem, state)
  stack.push(state)
  for one_succ in succ:
      if one_succ[0] in stack.list:
          continue
      else:
          if DFS(problem, stack, one_succ[0]) == "Goal":
              return "Goal"
  stack.pop()

def DFS2(problem, stack, state, rem):
    stack.push((state, []))

    while not stack.isEmpty():
        node, direction = stack.pop()
        for location, Dir, cost in getSuccessors(problem, node):
            if not location in rem:
                tmp = direction[:]
                tmp.append(Dir)
                if checkGoal(problem, location):
                    return tmp
                stack.push((location, tmp))
                rem.append(location)
    return []

def BFS(problem, queue, state, rem):
    queue.push((state, []))

    while not queue.isEmpty():
        node, direction = queue.pop()
        for location, Dir, cost in getSuccessors(problem, node):
            if not location in rem:
                tmp = direction[:]
                tmp.append(Dir)
                if checkGoal(problem, location):
                    return tmp
                queue.push((location, tmp))
                rem.append(location)
    return []

    "Lib. End"

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first [p 74].
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm [Fig. 3.18].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """
  "*** YOUR CODE HERE ***"
  start = problem.getStartState()
  stack = util.Stack()
  rem = []
  #DFS(problem, stack, start)
  return DFS2(problem, stack, start, rem)

def breadthFirstSearch(problem):
  "Search the shallowest nodes in the search tree first. [p 74]"
  "*** YOUR CODE HERE ***"
  start = problem.getStartState()
  queue = util.Queue()
  rem = []
  return BFS(problem, queue, start, rem)

      
def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  "*** YOUR CODE HERE ***"
  queue = util.PriorityQueue()
  start = problem.getStartState()
  queue.push((start, []), 0)
  rem = []
  while not queue.isEmpty():
      node, direction = queue.pop()
      for location, Dir, cost in getSuccessors(problem, node):
          if not location in rem:
              tmp = direction[:]
              tmp.append(Dir)
              if checkGoal(problem, location):
                  return tmp
              queue.push((location, tmp), problem.getCostOfActions(tmp))
              rem.append(location)
  return []

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  "*** YOUR CODE HERE ***"
  queue = util.PriorityQueue()
  start = problem.getStartState()
  queue.push((start, []), 0)
  rem = []
  while not queue.isEmpty():
      node, direction = queue.pop()
      for location, Dir, cost in getSuccessors(problem, node):
          if not location in rem:
              tmp = direction[:]
              tmp.append(Dir)
              if checkGoal(problem, location):
                  return tmp
              queue.push((location, tmp), problem.getCostOfActions(tmp) + heuristic(location, problem))
              rem.append(location)
  return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
