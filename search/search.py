# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    visited = set()
    s = util.Stack()
    direction_map = {}

    start, target = problem.getStartState(), None

    # If already at the target, no need to do anything
    if problem.isGoalState(start):
        return []

    # Populate stack with the initial coordinates
    s.push((start, None, 0))

    while not s.isEmpty():
        coord, direction, step = s.pop()

        # Have we found the food ?
        if problem.isGoalState(coord):
            target = (coord, direction)
            break

        if coord not in visited:
            visited.add(coord)
        
        for successor in problem.getSuccessors(coord):
            new_coord = successor[0]
            if new_coord not in visited:
                s.push(successor)
                direction_map[new_coord] = (coord, direction)

    path = []
    curr, direction = target[0], target[1]
    while curr and direction:
        path.append(direction)
        curr, direction = direction_map.get(curr, (None, None))
    
    path = list(reversed(path))
    return path

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    visited = set()
    q = util.Queue()

    start, target = problem.getStartState(), None

    # If already at the target, no need to do anything
    if problem.isGoalState(start):
        return []

    # Populate stack with the initial coordinates
    q.push((start, []))
    
    while not q.isEmpty():
        coord, path = q.pop()
        visited.add(coord)

        # Have we found the food ?
        if problem.isGoalState(coord):
            return path

        for successor in problem.getSuccessors(coord):
            adj = successor[0]
            adj_dir = successor[1]
            
            if adj not in visited:
                q.push((adj, path + [adj_dir]))

                visited.add(adj)
    
    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
