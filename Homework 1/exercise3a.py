import numpy as np
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

def montecarlo(numTries, depth, R, listNodes):
    startNode = listNodes
    for x in range(len(R)):
        if x == 5 or x == 11:
            print("Winner")
        else:
            print(x)
            for x in range(numTries):
                currNode = startNode
                for y in range(depth):
                    deptNode = currNode
                    for actions in range(4):
                        print(x, y, actions)


    #Start at a random position not including 5 or 11
    #go a maximum of depth steps
    #if it reaches node 11 or performs depth number of actions, add -1 to the R matrix
    #If it reaches node 5, add +1 to the R matrix

R = np.zeros([len(nodeList), 4])
print(R.shape)
montecarlo(1000, 10, R, nodeList)