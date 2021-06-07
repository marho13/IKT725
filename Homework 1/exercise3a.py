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

def search(depth, R, listNodes):
    for x in range(len(R)):
        if x == 5 or x == 11:
            print("Winner")
        else:
            sum = []
            currNode = listNodes[x]
            neighbours = currNode.neighbour
            for n in neighbours:
                if type(n) != type(None):
                    sum.append(0)
                    neighbourList = n.neighbour
                    for d in range(depth):
                        neighbourList = breadthFirst(neighbourList)
                        for output in neighbourList:
                            if type(output) == int:
                                # print(output)
                                if output == 1:
                                    sum[-1] += 5
                                else:
                                    sum[-1] += -1
                            else:
                                sum[-1] += -1

                else:
                    sum.append(-99999999999)
            print(sum)
            print()




    #Start at a random position not including 5 or 11
    #go a maximum of depth steps
    #if it reaches node 11 or performs depth number of actions, add -1 to the R matrix
    #If it reaches node 5, add +1 to the R matrix


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

R = np.zeros([len(nodeList), 4])
print(R.shape)
search(10, R, nodeList)
#the reward can be -1, 0 or 1 and should return