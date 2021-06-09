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

    return outputList

nodeList = createNodes(restrictions)

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

            print(sum)
            print()
            R = loopingSumandR(sum, R, x)

    print(R)



def breadthFirst(listOfInputs):
    outputList = []
    for a in range(len(listOfInputs)):
        if type(listOfInputs[a]) == type(node):
            neighbours = listOfInputs[a].neighbour
            for n in neighbours:
                if n == None:
                    outputList.append(listOfInputs[a])

                elif n.num == 5:
                    outputList.append(1)

                elif n.num == 11:
                    outputList.append(-1)

                else:
                    outputList.append(n)

        else:
            if listOfInputs[a] == 1:
                outputList.append(1)
            else:
                outputList.append(-1)
    return outputList

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



R = np.zeros([len(nodeList), 4])
print(R.shape)
search(10000, 10, R, nodeList) #Task done
