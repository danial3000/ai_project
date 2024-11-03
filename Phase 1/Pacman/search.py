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


def depthFirstSearch(problem):
    q = util.Stack()
    empty_action_list = []
    visited = set()
    q.push((problem.getStartState(), empty_action_list))
    visited.add(problem.getStartState())
    while not q.isEmpty():
        current_node, list_of_actions = q.pop()
        if problem.isGoalState(current_node):
            return list_of_actions
        for info in problem.getSuccessors(current_node):
            successor, action, step_cost = info
            if successor not in visited:
                new_list = list_of_actions + [action]
                q.push((successor, new_list))
                visited.add(successor)

    return empty_action_list


def breadthFirstSearch(problem):
    q = util.Queue()
    empty_action_list = []
    visited = set()
    q.push((problem.getStartState(), empty_action_list))
    visited.add(problem.getStartState())
    while not q.isEmpty():
        current_node, list_of_actions = q.pop()
        if problem.isGoalState(current_node):
            return list_of_actions
        for info in problem.getSuccessors(current_node):
            successor, action, step_cost = info
            if successor not in visited:
                new_list = list_of_actions + [action]
                q.push((successor, new_list))
                visited.add(successor)

    return empty_action_list


def uniformCostSearch(problem):

    priority_queue = util.PriorityQueue()
    start_state = problem.getStartState()
    priority_queue.push((start_state, []), 0)
    visited = set()

    while not priority_queue.isEmpty():
        state, actions = priority_queue.pop()

        if problem.isGoalState(state):
            return actions

        if state not in visited:
            visited.add(state)

            successors = problem.getSuccessors(state)
            for successor, action, cost in successors:
                if successor not in visited:
                    new_cost = problem.getCostOfActions(actions + [action])
                    priority_queue.push((successor, actions + [action]), new_cost)

    return []


def nullHeuristic(state, problem=None):
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    from searchAgents import cornersHeuristic
    heuristic = cornersHeuristic
    priority_queue = util.PriorityQueue()
    start_state = problem.getStartState()
    priority_queue.push((start_state, []), 0)
    visited = set()

    while not priority_queue.isEmpty():
        state, actions = priority_queue.pop()

        if problem.isGoalState(state):
            return actions

        if state not in visited:
            visited.add(state)

            successors = problem.getSuccessors(state)
            for successor, action, cost in successors:
                if successor not in visited:
                    new_cost = problem.getCostOfActions(actions + [action])
                    priority_queue.push((successor, actions + [action]), new_cost + heuristic(successor, problem))

    return []

'''
def cornersHeuristic(state, problem):
    position, corner = state
    corners = problem.corners
    unseens = [corner1 for i, corner1 in enumerate(corners) if not corner[i]]

    if len(unseens) == 0:
        return 0
    sum1 = 0
    for current_corner in unseens:
        sum1 = sum1 + util.manhattanDistance(position, current_corner)
    return sum1 * len(unseens) * 2
'''


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
