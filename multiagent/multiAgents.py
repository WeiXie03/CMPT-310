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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        food_dists = []
        for x in range(newFood.width):
            for y in range(newFood.height):
                if newFood[x][y]:
                    dist = manhattanDistance(newPos, (x,y))
                    food_dists.append(dist)
        # print("food dists: ", food_dists)
        if len(food_dists) > 0 and min(food_dists) > 0:
            min_food_dist = min(food_dists)
            # encourage Pacman to get within eating range
            avg_side_len = float((newFood.width + newFood.height)/2)
            # hyp = float(manhattanDistance((0,0), (newFood.width, newFood.height)))
            # norm_food_dists = float(min_food_dist) / hyp
            norm_food_dists = float(min_food_dist) / avg_side_len
        else:
            # will finish all the food
            norm_food_dists = -2
        norm_food_dists = 1 - norm_food_dists
        # print("food dist: ", norm_food_dists)
        # print("where food: ", newFood)

        delta_eat = 2 * (currentGameState.getNumFood() - len(food_dists))
        # print("delta eat: ", delta_eat)

        new_caps_grid = successorGameState.getCapsules()
        capsule_dists = []
        for (x,y) in new_caps_grid:
            dist = manhattanDistance(newPos, (x,y))
            capsule_dists.append(dist)
        # print("capsule dists: ", capsule_dists)
        if len(capsule_dists) > 0 and min(capsule_dists) > 0:
            min_capsule_dist = min(capsule_dists)
            # encourage Pacman to get within eating range
            norm_capsule_dists = float(min_capsule_dist)
        else:
            # will finish all the capsule
            norm_capsule_dists = -2
        norm_capsule_dists = 1 - norm_capsule_dists
        delta_eat_capsules = 2 * len(currentGameState.getCapsules()) - len(successorGameState.getCapsules())
        
        # Don't think about ghosts if not "threatening"
        GHOST_DIST_THRESH = 1
        norm_ghost_dists = 0
        for ghost in newGhostStates:
            ghost_dist = manhattanDistance(newPos, ghost.getPosition())
            if ghost_dist == 0:
                return -float("inf")
            elif ghost_dist < GHOST_DIST_THRESH:
                # mean squared error to "reflex" away from death
                norm_dist = float(GHOST_DIST_THRESH - ghost_dist) / float(GHOST_DIST_THRESH)
                norm_ghost_dists += norm_dist**2
        # encourage get farther from ghosts
        # mean avg
        norm_ghost_dists = float(norm_ghost_dists) / float(len(newGhostStates))
        # print("ghost dist: ", norm_ghost_dists)

        # discourage staying in place
        val = delta_eat + delta_eat_capsules + norm_food_dists + norm_capsule_dists + norm_ghost_dists
        if newPos == currentGameState.getPacmanPosition():
            return val - 0.5
        return val

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

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether5 or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        cur_score = -float("inf")
        num_ghosts = gameState.getNumAgents() - 1

        def miner(gameState, depth, agent_ind):
            val = float("inf")

            # end of game
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            action_opts = gameState.getLegalActions(agent_ind)
            for action in action_opts:
                if agent_ind == num_ghosts:
                    # last ghost, Pacman is index 0
                    val = min(val, maxer(gameState.generateSuccessor(agent_ind, action), depth))
                else:
                    val = min(val, miner(gameState.generateSuccessor(agent_ind, action), depth, agent_ind+1))
            return val
        
        def maxer(gameState, depth):
            val = -float("inf")
            
            # end of game, also at arbitrary defined max depth
            if gameState.isWin() or gameState.isLose() or (depth+1) == self.depth:
                return self.evaluationFunction(gameState)

            pac_action_opts = gameState.getLegalActions(0)
            for action in pac_action_opts:
                val = max(val, miner(gameState.generateSuccessor(0, action), depth+1, 1))
            return val
        
        pac_action_opts = gameState.getLegalActions(0)
        for action in pac_action_opts:
            score = miner(gameState.generateSuccessor(0, action), 0, 1)
            if score > cur_score:
                cur_score = score
                choose_action = action
        return choose_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        NUM_AGENTS = gameState.getNumAgents()
        MAX_DEPTH = self.depth

        def miner(gameState, depth, agent_ind, alpha, beta):
            actopts = {}
            for action in gameState.getLegalActions(agent_ind):
                val = state_val(gameState.generateSuccessor(agent_ind, action), depth, agent_ind+1, alpha, beta)[0]
                actopts[val] = action

                # Max parent won't let happen
                if val < alpha:
                    return (val, action)

                beta = min(beta, val)

            # min(a dict) is the min key
            return (min(actopts), actopts[min(actopts)])
        
        def maxer(gameState, depth, agent_ind, alpha, beta):
            actopts = {}
            for action in gameState.getLegalActions(agent_ind):
                val = state_val(gameState.generateSuccessor(agent_ind, action), depth, agent_ind+1, alpha, beta)[0]
                actopts[val] = action

                # Min parent won't let happen
                if val > beta:
                    return (val, action)

                alpha = max(alpha, val)

            # max(a dict) is the min key
            return (max(actopts), actopts[max(actopts)])
        
        def state_val(gameState, depth, agent_ind, alpha=-float("inf"), beta=float("inf")):
            if gameState.isWin() or gameState.isLose() or depth == MAX_DEPTH:
                return (self.evaluationFunction(gameState), "Stop")
            
            print("before {}, agent {}".format(depth, agent_ind))
            # after all agents gone once, Pacman next
            agent_ind = agent_ind % NUM_AGENTS
            # increment depth
            if agent_ind == NUM_AGENTS-1:
                depth += 1
            print("after {}, agent {}".format(depth, agent_ind))

            # score is Pacman's score
            if agent_ind == 0:
                return maxer(gameState, depth, agent_ind, alpha, beta)
            else:
                return miner(gameState, depth, agent_ind, alpha, beta)

        return state_val(gameState, 0, 0)[1]

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
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    food_dists = []
    for x in range(newFood.width):
        for y in range(newFood.height):
            if newFood[x][y]:
                dist = manhattanDistance(newPos, (x,y))
                food_dists.append(dist)
    # print("food dists: ", food_dists)
    if len(food_dists) > 0 and min(food_dists) > 0:
        min_food_dist = min(food_dists)
        # encourage Pacman to get within eating range
        avg_side_len = float((newFood.width + newFood.height)/2)
        # hyp = float(manhattanDistance((0,0), (newFood.width, newFood.height)))
        # norm_food_dists = float(min_food_dist) / hyp
        norm_food_dists = float(min_food_dist) / avg_side_len
    else:
        # will finish all the food
        norm_food_dists = -2
    norm_food_dists = 1 - norm_food_dists
    # print("food dist: ", norm_food_dists)
    # print("where food: ", newFood)

    delta_eat = 2 * (currentGameState.getNumFood() - len(food_dists))
    # print("delta eat: ", delta_eat)

    new_caps_grid = successorGameState.getCapsules()
    capsule_dists = []
    for (x,y) in new_caps_grid:
        dist = manhattanDistance(newPos, (x,y))
        capsule_dists.append(dist)
    # print("capsule dists: ", capsule_dists)
    if len(capsule_dists) > 0 and min(capsule_dists) > 0:
        min_capsule_dist = min(capsule_dists)
        # encourage Pacman to get within eating range
        norm_capsule_dists = float(min_capsule_dist)
    else:
        # will finish all the capsule
        norm_capsule_dists = -2
    norm_capsule_dists = 1 - norm_capsule_dists
    delta_eat_capsules = 2 * len(currentGameState.getCapsules()) - len(successorGameState.getCapsules())
    
    # Don't think about ghosts if not "threatening"
    GHOST_DIST_THRESH = 1
    norm_ghost_dists = 0
    for ghost in newGhostStates:
        ghost_dist = manhattanDistance(newPos, ghost.getPosition())
        if ghost_dist == 0:
            return -float("inf")
        elif ghost_dist < GHOST_DIST_THRESH:
            # mean squared error to "reflex" away from death
            if ghost.scaredTimer > 0:
                norm_dist = float(ghost_dist) / float(GHOST_DIST_THRESH)
            else:
                norm_dist = float(GHOST_DIST_THRESH - ghost_dist) / float(GHOST_DIST_THRESH)
            norm_ghost_dists += norm_dist**2
    # encourage get farther from ghosts
    # mean avg
    norm_ghost_dists = float(norm_ghost_dists) / float(len(newGhostStates))
    # print("ghost dist: ", norm_ghost_dists)

    # discourage staying in place
    val = delta_eat + delta_eat_capsules + norm_food_dists + norm_capsule_dists + norm_ghost_dists
    if newPos == currentGameState.getPacmanPosition():
        return val - 0.5
    return val

# Abbreviation
better = betterEvaluationFunction
