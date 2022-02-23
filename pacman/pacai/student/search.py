"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.queue import Queue
from pacai.util.stack import Stack
from pacai.util.priorityQueue import PriorityQueue

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***

    # checking if the starting state is the goal state
    stack = Stack()
    fringe = []
    visitedStates = []

    # if starting state is the goal state, return an empty list
    if problem.isGoal(problem.startingState()):
        return []

    # state space/type space is tuple with ((position, actionTaken, cost), path taken so far)
    node = ((problem.startingState(), None, 0), [])
    stack.push(node)
    fringe.append(problem.startingState())
    visitedStates.append(problem.startingState())

    while not stack.isEmpty():
        # for reference, state[0] = position (x,y), state[1] = action, state[2] = cost
        state, dir = stack.pop()
        fringe.remove(state[0])

        # iterating through all the successor states
        for child in problem.successorStates(state[0]):
            # copying path from parent state and copying it onto child states
            newDir = dir.copy()
            newDir.append(child[1])
            # if child turns out to be goal, end and return path
            if(problem.isGoal(child[0])):
                return newDir
            # if child has not been explored yet and not waiting to be explored
            if child[0] not in visitedStates and child[0] not in fringe:
                visitedStates.append(child[0])
                stack.push((child, newDir))
                fringe.append(child[0])

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***

    # very similar to Depth First Search, but instead of stack we utilize a queue
    queue = Queue()
    fringe = []
    visitedStates = []

    # if starting state is the goal state, return an empty list
    if problem.isGoal(problem.startingState()):
        return []

    # state space/type space is tuple with ((position, actionTaken, cost), path taken so far)
    node = ((problem.startingState(), None, 0), [])
    queue.push(node)
    fringe.append(problem.startingState())
    visitedStates.append(problem.startingState())

    while not queue.isEmpty():
        # for reference, state[0] = position (x,y), state[1] = action, state[2] = cost
        state, dir = queue.pop()
        fringe.remove(state[0])

        for child in problem.successorStates(state[0]):
            # copying path from parent state and copying it onto child states
            newDir = dir.copy()
            newDir.append(child[1])
            # if child turns out to be goal, end and return path
            if(problem.isGoal(child[0])):
                return newDir
            # if child has not been explored yet and not waiting to be explored
            if child[0] not in visitedStates and child[0] not in fringe:
                visitedStates.append(child[0])
                queue.push((child, newDir))
                fringe.append(child[0])

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***

    # very similar to Breadth First Search, but instead of queue we utilize a Priority Queue
    # with our priority defined by the action costs of the path
    queue = PriorityQueue()
    fringe = []
    visitedStates = []

    # if starting state is the goal state, return an empty list
    if problem.isGoal(problem.startingState()):
        return []

    # state space/type space is tuple with ((position, actionTaken, cost), path taken so far)
    node = ((problem.startingState(), None, 0), [])
    queue.push(node, 0)
    fringe.append(problem.startingState())
    visitedStates.append(problem.startingState())

    while not queue.isEmpty():
        # for reference, state[0] = position (x,y), state[1] = action, state[2] = cost
        state, dir = queue.pop()
        fringe.remove(state[0])

        for child in problem.successorStates(state[0]):
            # copying path from parent state and copying it onto child states
            newDir = dir.copy()
            newDir.append(child[1])
            # if child turns out to be goal, end and return path
            if(problem.isGoal(child[0])):
                return newDir
            # if child has not been explored yet and not waiting to be explored
            if child[0] not in visitedStates and child[0] not in fringe:
                visitedStates.append(child[0])
                # pushing to Priority Queue with action cost as priority
                queue.push((child, newDir), problem.actionsCost(newDir))
                fringe.append(child[0])

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***

    # same as UniformCostSearch, however its instead a combination of
    # both actionsCost and heuristic value combined
    queue = PriorityQueue()
    fringe = []
    visitedStates = []

    # if starting state is the goal state, return an empty list
    if problem.isGoal(problem.startingState()):
        return []

    # state space/type space is tuple with ((position, actionTaken, cost), path taken so far)
    node = ((problem.startingState(), None, 0 + heuristic(problem.startingState(), problem)), [])
    queue.push(node, 0)
    fringe.append(problem.startingState())
    visitedStates.append(problem.startingState())

    while not queue.isEmpty():
        # for reference, state[0] = position (x,y), state[1] = action, state[2] = cost
        state, dir = queue.pop()
        fringe.remove(state[0])
        for child in problem.successorStates(state[0]):
            # copying path from parent state and copying it onto child states
            newDir = dir.copy()
            newDir.append(child[1])
            # if child turns out to be goal, end and return path
            if(problem.isGoal(child[0])):
                return newDir
            # if child has not been explored yet and not waiting to be explored
            if child[0] not in visitedStates and child[0] not in fringe:
                # calculating the values with heurisitics
                heu = heuristic(child[0], problem)
                visitedStates.append(child[0])
                # pushing to Priority Queue with action cost AND heurisitic as priority
                queue.push((child, newDir), problem.actionsCost(newDir) + heu)
                fringe.append(child[0])
