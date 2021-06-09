import numpy as np
import random
left = 0
right = 1
up = 2
down = 3
restrictions = [[left, up], [down, up], [up], [up], [up, right], [], [right, up], [left, down], [], [right], [left],
                [], [right, up], [left, down], [right], [left], [right, up],
                [left], [up], [right], [left, down], [down], [down], [down], [right, down]]

Winblock = 5
LooseBlock = 11

winLoss = 5 #The reward/penalty

class node:
    def __init__(self, num):
        self.num = num
        self.reward = -0.1
        self.neighbour = []

    def setReward(self, reward):
        self.reward = reward

    def setNeighbour(self, left, right, up, down):
        self.neighbour.append(left)
        self.neighbour.append(right)
        self.neighbour.append(up)
        self.neighbour.append(down)

    def performAction(self, action):
        if self.neighbour[action] == None:
            return self
        else:
            return self.neighbour[action]

def createNodes(restrictions):
    lrud = [left, right, up, down]
    outputList = []
    for n in range(len(restrictions)):
        newNode = node(n)
        if n == Winblock:
            newNode.setReward(winLoss)
        elif n == LooseBlock:
            newNode.setReward(-winLoss)
        outputList.append(newNode)

    for tile in range(len(outputList)):
        neighbour = []
        for direction in lrud:
            # print(direction)
            if direction in restrictions[tile]:
                neighbour.append(None)
            else:
                if direction == 0:
                    neighbour.append(outputList[tile-1])

                if direction == 1:
                    neighbour.append(outputList[tile+1])

                if direction == 2:
                    neighbour.append(outputList[tile-5])

                if direction == 3:
                    neighbour.append(outputList[tile+5])
        outputList[tile].setNeighbour(neighbour[0], neighbour[1], neighbour[2], neighbour[3])
        print("Node {} has neighbours:".format(tile))
        for ne in neighbour:
            if type(ne) != type(None):
                print(ne.num, end=" ")
        print()

    return outputList

def loopingSumandR(sum, R, numInR):
    first, second = -99999, -99999
    fInd, sInd = 0, 0
    for f in range(len(sum)):
        if sum[f] > first:
            first = sum[f]
            fInd = f
    for s in range(len(sum)):
        if s != fInd:
            if sum[s] > second:
                second = sum[s]
                sInd = s

    for x in range(len(sum)):
        if x != fInd or x != sInd:
            R[numInR][x] = -1
        R[numInR][fInd] = 1
        R[numInR][sInd] = 0
    return R

def search(numTries ,depth, R, listNodes):
    for x in range(len(R)):
        if x == 5:
            R[x][0] = 5
            R[x][1] = 5
            R[x][2] = 5
            R[x][3] = 5
        if x == 11:
            R[x][0] = -5
            R[x][1] = -5
            R[x][2] = -5
            R[x][3] = -5

        else:
            sum = []
            currNode = listNodes[x]
            neighbours = currNode.neighbour
            for n in neighbours:
                sum.append(0)
                for _ in range(numTries):
                    if type(n) != type(None):
                        output = monteCarlo(depth, 1, n)
                    else:
                        output = monteCarlo(depth, 1, currNode)
                    sum[-1] += output

            # print(sum)
            # print()
            R = loopingSumandR(sum, R, x)

    print(R)
    return R

def monteCarlo(maxDepth, depth, node):
    currNode = node
    if node == nodeList[5]:
        return 1
    elif node == nodeList[11]:
        return -1

    for i in range(depth, maxDepth):
        neighbours = currNode.neighbour
        randNum = random.randint(0, len(neighbours)-1)
        nextNode = neighbours[randNum]
        if nextNode == nodeList[5]:
            return 1
        elif nextNode == nodeList[11]:
            return -1

        if type(nextNode) != type(None):
            currNode = nextNode

    return 0

class policyIteration:
    def __init__(self, gamma, stateNum, actionNum, R, listNodes):
        #Task infomation
        self.gamma = gamma
        self.R = R
        self.nodes = listNodes
        self.currState = None
        #Creating the pi, V and R, the P is deterministic so we can remove it
        self.policy = [(0) for __ in range(stateNum)]
        self.valueEstimate = [[(0.0) for _ in range(actionNum)] for __ in range(stateNum)]
        #Number of states, number of actions
        self.numberOfStates = stateNum
        self.actionNum = actionNum
        #
        # Current node and his number
        #
        # print(len(self.policy), len(self.policy[0]))
        # print(self.policy)
        # print(self.valueEstimate)
        # #Environment to interact with


    # def randomState(): #Start with a random state or move through each of the nodes?

    def performAction(self, action):
        newNode = self.currState.neighbour[action] #Change this
        if type(newNode) != type(None):
            return newNode
        return self.currState

    def getAction(self, state):
        return self.policy[state]

    def calcPolicy(self):
        newPol = []
        for s in range(self.numberOfStates):
            newPol.append(0)
            maxVal = -9999
            maxInd = 0
            for action in range(self.actionNum):
                if self.valueEstimate[s][action] > maxVal:
                    maxVal = self.valueEstimate[s][action]
                    maxInd = action

            newPol[-1] = maxInd
        self.policy = newPol

    def CalculateValueEstimates(self):
        sumy = [0.0, 0.0, 0.0, 0.0]
        neighbour = self.currState.neighbour
        for action in range(self.actionNum):
            if type(neighbour[action]) != type(None):
                if neighbour[action].num == 5:
                    sumy[action] = (5.0*self.gamma) + self.R[self.currState.num][action]
                if neighbour[action].num == 11:
                    sumy[action] = (-5.0*self.gamma) + self.R[self.currState.num][action]

                else:
                    futureEstimate = self.gamma * self.getValueForState(neighbour[action].num)
                    sumy[action] = self.R[self.currState.num][action] + futureEstimate

            else:
                futureEstimate = self.getValueForState(self.currState.num)
                futureEstimate = self.gamma * futureEstimate
                sumy[action] = self.R[self.currState.num][action] + futureEstimate * 0.5

        self.valueEstimate[self.currState.num] = sumy

    def getValueForState(self, state):
        maxVal = -99999
        maxInd = 0
        for action in range(self.actionNum):
            if self.valueEstimate[state][action] > maxVal:
                maxVal = self.valueEstimate[state][action]
                maxInd = action
        return self.valueEstimate[state][maxInd]

    def training(self):
        for x in range(len(self.R)):
            if x != 5 and x != 11:
                for i in range(100):
                    self.currState = self.nodes[x]
                    for _ in range(15):
                        self.CalculateValueEstimates()
                        self.calcPolicy()
                        action = self.getAction(self.currState.num)
                        if type(self.currState.neighbour[action]) != type(None):
                            self.currState = self.currState.neighbour[action]
                        # self.improveValueEstimate(self.city, action, reward)
                        # self.improvePolicy()
                    # print("Iteration {}: gave Estimates {}".format(i, self.valueEstimate))
        print(self.policy)
        print(self.valueEstimate)

nodeList = createNodes(restrictions)
R = np.zeros([len(nodeList), 4])
# print(R.shape)
R = search(5000, 10, R, nodeList) #Next task run Value estimator
print("Done searching")

point8 = policyIteration(0.8, len(R), len(R[0]), R, nodeList)
one = policyIteration(1.0, len(R), len(R[0]), R, nodeList)

one.training()
print()
print()
point8.training()