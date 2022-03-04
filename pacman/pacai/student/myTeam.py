from queue import Empty
import random
import abc
import glob
import logging
import os
import time


from pacai.util import reflection
from pacai.agents.base import BaseAgent
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions
# from pacai.agents.capture.capture import CaptureAgent
from pacai.core import distanceCalculator
from pacai.util import util
from pacai.util import reflection
from pacai.util import probability
from random import choice

# ------ for eval function ------
from pacai.core import distance
from pacai.bin import pacman
# ------ for eval function ------

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.student.myTeam.DummyAgent',
        second = 'pacai.student.myTeam.DummyAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    # This is how we should call the agents in the file -J
    firstAgent = OffensiveReflexAgent
    secondAgent = DefensiveReflexAgent

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]
class CaptureAgent(BaseAgent):
    """
    A base class for capture agents.
    This class has some helper methods that students may find useful.

    The recommended way of setting up a capture agent is just to extend this class
    and implement `CaptureAgent.chooseAction`.
    """

    def __init__(self, index, timeForComputing = 0.1, **kwargs):
        super().__init__(index, **kwargs)

        # Whether or not you're on the red team
        self.red = None

        # Agent objects controlling you and your teammates
        self.agentsOnTeam = None

        # Maze distance calculator
        self.distancer = None

        # A history of observations
        self.observationHistory = []

        # Time to spend each turn on computing maze distances
        self.timeForComputing = timeForComputing

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.
        """

        self.red = gameState.isOnRedTeam(self.index)
        self.distancer = distanceCalculator.Distancer(gameState.getInitialLayout())

        self.distancer.getMazeDistances()

    def final(self, gameState):
        self.observationHistory = []

    def registerTeam(self, agentsOnTeam):
        """
        Fills the self.agentsOnTeam field with a list of the
        indices of the agents on your team.
        """

        self.agentsOnTeam = agentsOnTeam

    def getAction(self, gameState):
        """
        Calls `CaptureAgent.chooseAction` on a grid position, but continues on partial positions.
        If you subclass `CaptureAgent`, you shouldn't need to override this method.
        It takes care of appending the current state on to your observation history
        (so you have a record of the game states of the game) and will call your
        `CaptureAgent.chooseAction` method if you're in a proper state.
        """

        self.observationHistory.append(gameState)

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()

        if (myPos != util.nearestPoint(myPos)):
            # We're halfway from one position to the next.
            return gameState.getLegalActions(self.index)[0]
        else:
            return self.chooseAction(gameState)

    @abc.abstractmethod
    def chooseAction(self, gameState):
        """
        Override this method to make a good agent.
        It should return a legal action within the time limit
        (otherwise a random legal action will be chosen for you).
        """

        pass

    def getFood(self, gameState):
        """
        Returns the food you're meant to eat.
        This is in the form of a `pacai.core.grid.Grid`
        where `m[x][y] = True` if there is food you can eat (based on your team) in that square.
        """

        if (self.red):
            return gameState.getBlueFood()
        else:
            return gameState.getRedFood()

    def getFoodYouAreDefending(self, gameState):
        """
        Returns the food you're meant to protect (i.e., that your opponent is supposed to eat).
        This is in the form of a `pacai.core.grid.Grid`
        where `m[x][y] = True` if there is food at (x, y) that your opponent can eat.
        """

        if (self.red):
            return gameState.getRedFood()
        else:
            return gameState.getBlueFood()

    def getCapsules(self, gameState):
        if (self.red):
            return gameState.getBlueCapsules()
        else:
            return gameState.getRedCapsules()

    def getCapsulesYouAreDefending(self, gameState):
        if (self.red):
            return gameState.getRedCapsules()
        else:
            return gameState.getBlueCapsules()

    def getOpponents(self, gameState):
        """
        Returns agent indices of your opponents. This is the list of the numbers
        of the agents (e.g., red might be 1, 3, 5)
        """

        if self.red:
            return gameState.getBlueTeamIndices()
        else:
            return gameState.getRedTeamIndices()

    def getTeam(self, gameState):
        """
        Returns agent indices of your team. This is the list of the numbers
        of the agents (e.g., red might be the list of 1,3,5)
        """

        if (self.red):
            return gameState.getRedTeamIndices()
        else:
            return gameState.getBlueTeamIndices()

    def getScore(self, gameState):
        """
        Returns how much you are beating the other team by in the form of a number
        that is the difference between your score and the opponents score.
        This number is negative if you're losing.
        """

        if (self.red):
            return gameState.getScore()
        else:
            return gameState.getScore() * -1

    def getMazeDistance(self, pos1, pos2):
        """
        Returns the distance between two points using the builtin distancer.
        """

        return self.distancer.getDistance(pos1, pos2)

    def getPreviousObservation(self):
        """
        Returns the `pacai.core.gamestate.AbstractGameState` object corresponding to
        the last state this agent saw.
        That is the observed state of the game last time this agent moved,
        this may not include all of your opponent's agent locations exactly.
        """

        if (len(self.observationHistory) <= 1):
            return None

        return self.observationHistory[-2]

    def getCurrentObservation(self):
        """
        Returns the GameState object corresponding this agent's current observation
        (the observed state of the game - this may not include
        all of your opponent's agent locations exactly).

        Returns the `pacai.core.gamestate.AbstractGameState` object corresponding to
        this agent's current observation.
        That is the observed state of the game last time this agent moved,
        this may not include all of your opponent's agent locations exactly.
        """

        if (len(self.observationHistory) == 0):
            return None

        return self.observationHistory[-1]

class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at `pacai.core.baselineTeam` for more details about how to create an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """

        super().registerInitialState(gameState)

        # Your initialization code goes here, if you need any.

    def chooseAction(self, gameState):
        """
        Randomly pick an action.
        """

        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.depth = 2

    # MAIN ACTION FUNCTION
    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """

        # ORIGINAL IMPLEMENTATION 

        # actions = gameState.getLegalActions(self.index)

        # start = time.time()
        # values = [self.evaluate(gameState, a) for a in actions]
        # logging.debug('evaluate() time for agent %d: %.4f' % (self.index, time.time() - start))

        # maxValue = max(values)
        # bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        # return random.choice(bestActions)

        #IMPLEMENTING WITH JUST EVAL FUNCTIONS
        # actions = gameState.getLegalActions(self.index)

        # if "Stop" in actions:
        #     actions.remove("Stop")

        # start = time.time()
        # values = [self.evalFunction(gameState.generateSuccessor(self.index, a)) for a in actions]
        # logging.debug('evaluate() time for agent %d: %.4f' % (self.index, time.time() - start))

        # maxValue = max(values)
        # print("values: ", values, "MAXVALUE: ", maxValue)
        # bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        # print(bestActions)

        # # if "Stop" in bestActions:
        # #     bestActions.remove("Stop")

        # return random.choice(bestActions)


        # EXPECTIMINIMAX IMPLEMENTATION
        action = self.getActionExpectiminimax(gameState)
        print(action)
        return action

        # Q LEARNING IMPLEMENTATION
        # return self.getActionQLearning(gameState)
    
    # EXPECTIMINIMAX IMPLEMENTATION
    def getActionExpectiminimax(self, gameState):
        # sentinal value
        max = -9999999
        returnAction = ''
        # print(gameState.getLegalActions())
        for action in gameState.getLegalActions(self.index):
            # skip stop action
            if action == Directions.STOP:
                continue
            # print(action, gameState.getLegalActions(self.index))
            newState = gameState.generateSuccessor(self.index, action)
            agentIndex = newState.getLastAgentMoved()
            checkVal = self.expectiMininmax(
                agentIndex + 1, newState, self.index, self.depth, True)
            # finding the max value out of all actions
            if checkVal > max:
                max = checkVal
                returnAction = action
        # print(returnAction)
        return returnAction

    # expectiminimax helper function
    def expectiMininmax(self, agentIndex, gameState, depth, maxDepth, chance):
        if gameState.isOver() or depth == maxDepth:
            # print("here")
            return self.evalFunction(gameState)
        
        # chance only if agentIndex is self.index
        if not chance:
            if agentIndex == self.index:
                maxVal = -999999
                for action in gameState.getLegalActions(agentIndex):
                    # skips the stop action
                    if action == Directions.STOP:
                        continue
                    newState = gameState.generateSuccessor(agentIndex, action)
                    # for this section only finding the max value
                    maxVal = max(maxVal, self.expectiMininmax(
                        agentIndex + 1, newState, depth, maxDepth, True))
                return maxVal
        else:
            # dependent on two possible states
            # either value is less or more
            if agentIndex == gameState.getNumAgents() - 1:
                averageVal = 0
                # averaging all the possible values
                # node right before a max node
                for action in gameState.getLegalActions(agentIndex):
                    newState = gameState.generateSuccessor(agentIndex, action)
                    averageVal += self.expectiMininmax(
                        0, newState, depth + 1, maxDepth, False)
                return averageVal / len(gameState.getLegalActions(agentIndex))
            elif agentIndex % gameState.getNumAgents() != 0:
                averageVal = 0
                # averaging all the possible values
                # node before more min nodes
                for action in gameState.getLegalActions(agentIndex):
                    newState = gameState.generateSuccessor(agentIndex, action)
                    averageVal += self.expectiMininmax(
                        agentIndex + 1, newState, depth, maxDepth, True)
                return averageVal / len(gameState.getLegalActions(agentIndex))

    # Q LEARNING IMPLEMENTATION
    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """
        # check if state action pair in q dictionary already
        if (state, action) not in self.q.keys():
            self.q[(state, action)] = 0

        # return the recorded qVal
        return self.q[(state, action)]

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """

        # check for terminal state
        if state == 'TERMINAL_STATE':
            return 0.0

        # check if state has list of legal actions
        if len(self.getLegalActions(state)) == 0:
            return 0.0

        # get optimal action and return the state action pair
        mainAction = self.getPolicy(state)
        return self.getQValue(state, mainAction)

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        listOfActions = self.getLegalActions(state)

        # check if there any legal actions for state
        if not listOfActions:
            return None

        # recording all actions in list
        mainList = []
        for action in listOfActions:
            mainList.append((self.getQValue(state, action), action))

        # getting the maxValue of the list
        return max(mainList, key=lambda val: val[0])[1]

    def getActionQLearning(self, state):
        # set up randomness of getting next action
        randVal = probability.flipCoin(1 - self.getEpsilon())
        allActions = self.getLegalActions(state)

        # if it has that value return optimal, else return otherwise
        if randVal:
            return self.getPolicy(state)
        else:
            return choice(allActions)

    def update(self, state, action, nextState, reward):
        # set up of variable names
        gamma = self.getGamma()
        alpha = self.getAlpha()
        maxQ = self.getValue(nextState)

        # set up of sample
        sample = reward + gamma * maxQ
        # getting QValue
        qVal = self.getQValue(state, action)
        # calculating qValue
        self.q[(state, action)] = (1 - alpha) * qVal + (alpha * sample)

    @abc.abstractmethod
    def evalFunction(self, state):
        """
        Eval Function to define best state possible, defined differently
        depending on offensive or defensive
        """
        pass

    # PRE IMPLEMENTED FUNCTIONS
    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if (pos != util.nearestPoint(pos)):
            # Only half a grid position was covered.
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights.
        """

        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        stateEval = sum(features[feature] * weights[feature] for feature in features)

        return stateEval

    def getFeatures(self, gameState, action):
        """
        Returns a dict of features for the state.
        The keys match up with the return from `ReflexCaptureAgent.getWeights`.
        """

        successor = self.getSuccessor(gameState, action)

        return {
            'successorScore': self.getScore(successor)
        }

    def getWeights(self, gameState, action):
        """
        Returns a dict of weights for the state.
        The keys match up with the return from `ReflexCaptureAgent.getFeatures`.
        """

        return {
            'successorScore': 1.0
        }

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food.
    This agent will give you an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        features = {}
        print(gameState.getLegalActions(self.index))
        print(gameState.getNumAgents())
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        return features

    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            'distanceToFood': -1
        }

    def evalFunction(self, state):
        # oldState = self.getPreviousObservation()
        # if oldState == None:
        #     return 0
        # prevPosition = oldState.getAgentState(self.index).getPosition() # prev pos of pacman
        # currPosition = state.getAgentState(self.index).getPosition()  # current position of pacman
        # oldFood = self.getFood(state).asList()  # list of foods to eat @ curr state
        
        # # Getting location of previous state's food
        # if (oldState != None):
        #     oldoldFood = self.getFood(oldState).asList()
        # else:
        #     oldoldFood = oldFood

        # oldGhostStates = []  # list curr positions of the ghosts
        # oldGhostStates2 = []
        # for agent in self.getOpponents(state):
        #     oldGhostStates.append(state.getAgentState(agent).getPosition())
        #     oldGhostStates2.append(state.getAgentState(agent))

        # addedScore = 0  # score to be added
 
        # # store the position of the closest food
        # closestFoodCoords = oldFood[0]
        # # distance b/w you & the closest food
        # foodDistance = self.getMazeDistance(currPosition, oldFood[0])

        # # go through list of all foods and find the one w/ the closest position
        # for i in oldFood:
        #     if (self.getMazeDistance(currPosition, i) < foodDistance):
        #         closestFoodCoords = i  # store the position of the closest food
        #         foodDistance = self.getMazeDistance(currPosition, i)  # store the distance

        # if foodDistance != 0:
        #     addedScore += (1/foodDistance) * 20
        # else:
        #     addedScore += 100

        # if len(oldFood) < len(oldoldFood):
        #     addedScore += 100
        
        # # distance b/w you & the closest ghost
        # ghostDistance = self.getMazeDistance(currPosition, oldGhostStates[0])

        # # go through list of all ghosts and find the one w/ the closest position
        # for i in oldGhostStates:
        #     if (self.getMazeDistance(currPosition, i) < ghostDistance):
        #         ghostDistance = self.getMazeDistance(currPosition, i)  # store the distance
       
        # # go towards ghost if scared & close & enough time
        # # scaredTime = [ghostState.getScaredTimer() for ghostState in oldGhostStates2]
        # # scared = min(scaredTime)
        # # if (ghostDistance < 2):
        # #     if (scared < 2):
        # #         addedScore -= 10000
        # #     else:
        # #         addedScore += 10000
        
        # addedScore += ghostDistance * 2

        # #if you're stopped, keep losing points
        # if (currPosition == prevPosition):
        #     addedScore -= 50

        # return state.getScore() + + ghostDistance / (foodDistance * 7) + addedScore
        currPosition = state.getAgentState(self.index).getPosition()  # current position of pacman
        oldFood = self.getFood(state).asList()  # list of foods to eat @ curr state
        oldState = self.getPreviousObservation()

        if oldState == None:
            return 0

        # Getting location of previosu state's food
        if (oldState != None):
            oldoldFood = self.getFood(oldState).asList()
        else:
            oldoldFood = oldFood

        addedScore = 0  # score to be added
 
        # store the position of the closest food
        closestFoodCoords = oldFood[0]
        # distance b/w you & the closest food
        foodDistance = self.getMazeDistance(currPosition, oldFood[0])

        # go through list of all foods and find the one w/ the closest position
        for i in oldFood:
            if (self.getMazeDistance(currPosition, i) < foodDistance):
                closestFoodCoords = i  # store the position of the closest food
                foodDistance = self.getMazeDistance(currPosition, i)  # store the distance

        if foodDistance != 0:
            addedScore += (1/foodDistance) * 20
        else:
            addedScore += 100

        if len(oldFood) < len(oldoldFood):
            addedScore += 100
        
        oldGhostStates = []  # list curr positions of the ghosts
        oldGhostStates2 = []
        for agent in self.getOpponents(state):
            if state.getAgentState(agent).isGhost():
                oldGhostStates.append(state.getAgentState(agent).getPosition())
                oldGhostStates2.append(state.getAgentState(agent))

        # distance b/w you & the closest ghost
        ghostDistance = self.getMazeDistance(currPosition, oldGhostStates[0])

        # go through list of all ghosts and find the one w/ the closest position
        for i in oldGhostStates:
            if (self.getMazeDistance(currPosition, i) < ghostDistance):
                ghostDistance = self.getMazeDistance(currPosition, i)  # store the distance

        scaredTime = [ghostState.getScaredTimer() for ghostState in oldGhostStates2]
        scared = min(scaredTime)
        if (ghostDistance < 3):
            if (scared < 3):
                addedScore -= 10000
            # else:
            #     addedScore += 10000

        addedScore += (1/ghostDistance) * 5

        # if distance to food is smaller than the previous distance, add food points
        # if (distance.manhattan(newPosition, closestFoodCoords) < foodDistance):
        #     # aka, pacman is closer to food, increase points
        #     addedScore += pacman.FOOD_POINTS
        #     addedScore += pacman.FOOD_POINTS
        #     addedScore += pacman.FOOD_POINTS

        # if you're stopped, keep losing points
        # if (currPosition == newPosition):
        #     addedScore -= 50
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # add in stuff to make pacman go towards ghost when they're scared/ ate a pellet
        # useful stuff:
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        return state.getScore() + addedScore



class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that tries to keep its side Pacman-free.
    This is to give you an idea of what a defensive agent could be like.
    It is not the best or only way to make such an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        features = {}

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'stop': -100,
            'reverse': -2
        }

    def evalFunction(self, state):
        """
        you are the ghost & you're trying to defend your food
        if ghost:
            - go towards pacman to try to eat them
        if scared ghost:
            - run away from pacman
        """

        currPosition = state.getAgentState(self.index).getPosition()  # current position of us
        #oldFood = self.getFoodYouAreDefending(state).asList()  # list of foods to defend
        oldState = self.getPreviousObservation()  # old position of us

        if oldState == None:
            return 0

        addedScore = 0  # score to be added

        currState = []
        for agent in self.getTeam(state):
            if state.getAgentState(agent).isGhost():
                currState.append(state.getAgentState(agent))
        
        oldGhostStates = []  # list curr positions of the opp pacmen
        oldGhostStates2 = []  # list opp pacmen states
        for agent in self.getOpponents(state):
            if state.getAgentState(agent).isPacman():
                oldGhostStates.append(state.getAgentState(agent).getPosition())
                oldGhostStates2.append(state.getAgentState(agent))

        if (len(oldGhostStates2) != 0):
            # distance b/w you & the closest pacman
            pacDistance = self.getMazeDistance(currPosition, oldGhostStates[0])

            # go through list of all pacmen and find the one w/ the closest position
            for i in oldGhostStates:
                if (self.getMazeDistance(currPosition, i) < pacDistance):
                    pacDistance = self.getMazeDistance(currPosition, i)  # store the distance

            scaredTime = [scaredState.getScaredTimer() for scaredState in currState]
            scared = min(scaredTime)
            if (pacDistance < 3):
                if (scared < 3): # if pac is close and ur scared, decrease score
                    addedScore -= 10000
                else: # if pac is close and ur not scared, increase score
                    addedScore += 10000
            
            addedScore += (1/pacDistance) * 2 # smaller the distance, the more is added

            if (pacDistance == 0 and not currState.isScared()):
                addedScore += 100000000

        return state.getScore() + addedScore
        
class OffensiveReflexAgent2(ReflexCaptureAgent):
    """
    A reflex agent that seeks food.
    This agent will give you an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        features = {}
        print(gameState.getLegalActions(self.index))
        print(gameState.getNumAgents())
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        return features

    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            'distanceToFood': -1
        }

    def evalFunction(self, state):
        # oldState = self.getPreviousObservation()
        # if oldState == None:
        #     return 0
        # prevPosition = oldState.getAgentState(self.index).getPosition() # prev pos of pacman
        # currPosition = state.getAgentState(self.index).getPosition()  # current position of pacman
        # oldFood = self.getFood(state).asList()  # list of foods to eat @ curr state
        
        # # Getting location of previous state's food
        # if (oldState != None):
        #     oldoldFood = self.getFood(oldState).asList()
        # else:
        #     oldoldFood = oldFood

        # oldGhostStates = []  # list curr positions of the ghosts
        # oldGhostStates2 = []
        # for agent in self.getOpponents(state):
        #     oldGhostStates.append(state.getAgentState(agent).getPosition())
        #     oldGhostStates2.append(state.getAgentState(agent))

        # addedScore = 0  # score to be added
 
        # # store the position of the closest food
        # closestFoodCoords = oldFood[0]
        # # distance b/w you & the closest food
        # foodDistance = self.getMazeDistance(currPosition, oldFood[0])

        # # go through list of all foods and find the one w/ the closest position
        # for i in oldFood:
        #     if (self.getMazeDistance(currPosition, i) < foodDistance):
        #         closestFoodCoords = i  # store the position of the closest food
        #         foodDistance = self.getMazeDistance(currPosition, i)  # store the distance

        # if foodDistance != 0:
        #     addedScore += (1/foodDistance) * 20
        # else:
        #     addedScore += 100

        # if len(oldFood) < len(oldoldFood):
        #     addedScore += 100
        
        # # distance b/w you & the closest ghost
        # ghostDistance = self.getMazeDistance(currPosition, oldGhostStates[0])

        # # go through list of all ghosts and find the one w/ the closest position
        # for i in oldGhostStates:
        #     if (self.getMazeDistance(currPosition, i) < ghostDistance):
        #         ghostDistance = self.getMazeDistance(currPosition, i)  # store the distance
       
        # # go towards ghost if scared & close & enough time
        # # scaredTime = [ghostState.getScaredTimer() for ghostState in oldGhostStates2]
        # # scared = min(scaredTime)
        # # if (ghostDistance < 2):
        # #     if (scared < 2):
        # #         addedScore -= 10000
        # #     else:
        # #         addedScore += 10000
        
        # addedScore += ghostDistance * 2

        # #if you're stopped, keep losing points
        # if (currPosition == prevPosition):
        #     addedScore -= 50

        # return state.getScore() + + ghostDistance / (foodDistance * 7) + addedScore
        
        
        currPosition = state.getAgentState(self.index).getPosition()  # current position of pacman
        oldCapsules = self.getCapsules(state)  # list of foods to eat @ curr state
        oldState = self.getPreviousObservation()

        if oldState == None:
            return 0

        if len(oldCapsules) != 0:
            # Getting location of previosu state's food
            if (oldState != None):
                oldoldCapsules = self.getCapsules(oldState)
            else:
                oldoldCapsules = oldCapsules

            addedScore = 0  # score to be added
    
            # store the position of the closest food
            closestFoodCoords = oldCapsules[0]
            # distance b/w you & the closest food
            foodDistance = self.getMazeDistance(currPosition, oldCapsules[0])

            # go through list of all foods and find the one w/ the closest position
            for i in oldCapsules:
                if (self.getMazeDistance(currPosition, i) < foodDistance):
                    closestFoodCoords = i  # store the position of the closest food
                    foodDistance = self.getMazeDistance(currPosition, i)  # store the distance

            if foodDistance != 0:
                addedScore += (1/foodDistance) * 20
            else:
                addedScore += 100

            if len(oldCapsules) < len(oldoldCapsules):
                addedScore += 100
        
        
        
        ##=============================================================================
        
        
        else:
            oldFood = self.getFood(state).asList()  # list of foods to eat @ curr state
            oldState = self.getPreviousObservation()

            if oldState == None:
                return 0

            # Getting location of previosu state's food
            if (oldState != None):
                oldoldFood = self.getFood(oldState).asList()
            else:
                oldoldFood = oldFood

            addedScore = 0  # score to be added
    
            # store the position of the closest food
            closestFoodCoords = oldFood[0]
            # distance b/w you & the closest food
            foodDistance = self.getMazeDistance(currPosition, oldFood[0])

            # go through list of all foods and find the one w/ the closest position
            for i in oldFood:
                if (self.getMazeDistance(currPosition, i) < foodDistance):
                    closestFoodCoords = i  # store the position of the closest food
                    foodDistance = self.getMazeDistance(currPosition, i)  # store the distance

            if foodDistance != 0:
                addedScore += (1/foodDistance) * 20
            else:
                addedScore += 100

            if len(oldFood) < len(oldoldFood):
                addedScore += 100
        
        oldGhostStates = []  # list curr positions of the ghosts
        oldGhostStates2 = []
        for agent in self.getOpponents(state):
            if state.getAgentState(agent).isGhost():
                oldGhostStates.append(state.getAgentState(agent).getPosition())
                oldGhostStates2.append(state.getAgentState(agent))

        # distance b/w you & the closest ghost
        ghostDistance = self.getMazeDistance(currPosition, oldGhostStates[0])

        # go through list of all ghosts and find the one w/ the closest position
        for i in oldGhostStates:
            if (self.getMazeDistance(currPosition, i) < ghostDistance):
                ghostDistance = self.getMazeDistance(currPosition, i)  # store the distance

        scaredTime = [ghostState.getScaredTimer() for ghostState in oldGhostStates2]
        scared = min(scaredTime)
        if (ghostDistance < 3):
            if (scared < 3):
                addedScore -= 10000
            # else:
            #     addedScore += 10000

        addedScore += (1/ghostDistance) * 5

        # if distance to food is smaller than the previous distance, add food points
        # if (distance.manhattan(newPosition, closestFoodCoords) < foodDistance):
        #     # aka, pacman is closer to food, increase points
        #     addedScore += pacman.FOOD_POINTS
        #     addedScore += pacman.FOOD_POINTS
        #     addedScore += pacman.FOOD_POINTS

        # if you're stopped, keep losing points
        # if (currPosition == newPosition):
        #     addedScore -= 50
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # add in stuff to make pacman go towards ghost when they're scared/ ate a pellet
        # useful stuff:
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        return state.getScore() + addedScore