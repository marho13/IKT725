#!/usr/bin/env python
# coding=utf-8
import numpy as np
import random

a = 2
b = 1
c = -3
startCity = random.randint(0, 1)
gamma = 0.9
numberOfStates = 2
numberOfActions = 2
print(startCity)

class cityWork:
    def __init__(self, c, a, b, startCity):
        self.aRew = a
        self.bRew = b
        self.c = c
        self.currentCity = startCity


    def performActions(self, a):
        if a == 0:
            if self.currentCity == 0:
                self.currentCity = 1
                reward = self.c
            if self.currentCity == 1:
                self.currentCity = 0
                reward = self.c

        else:
            if self.currentCity == 0:
                reward = self.aRew
            else:
                reward = self.bRew
        return reward, self.currentCity

class policyIteration:
    def __init__(self, c, a, b, gamma, startCity, stateNum, actionNum):
        #Task infomation
        self.gamma = gamma
        self.c = c
        self.a = a
        self.b = b
        self.city = startCity
        #Creating the pi, V and R, the P is deterministic so we can remove it
        self.policy = [[0.5, 0.5], [0.5, 0.5]]
        self.valueEstimate = [[0.0, 0.0], [0.0, 0.0]]
        self.R = [[0, 0], [0, 0]]
        #Number of states, number of actions
        self.numberOfStates = stateNum
        self.actionNum = actionNum
        # #Environment to interact with
        # self.env = env
        #Create the R tables values, can also be estimated
        self.createReward()

    def createReward(self):
        self.R[0][1] = -3
        self.R[0][0] = 2
        self.R[1][0] = 1
        self.R[1][1] = -3

    def getAction(self, state):
        max = 0.0
        ind = 0
        for a in range(len(self.policy[state])):
            if self.policy[state][a] + ((random.randrange(0, 10, 1)-5)/20) > max:
                max = self.policy[state][a]
                ind = a
        return ind

    def improvePolicy(self):
        for s in range(self.numberOfStates):
            if self.valueEstimate[s][0] > self.valueEstimate[s][1]:
                self.policy[s][0] += 0.0005
                self.policy[s][1] -= 0.0005

            elif self.valueEstimate[s][0] < self.valueEstimate[s][1]:
                self.policy[s][0] -= 0.0005
                self.policy[s][1] += 0.0005

    def CalculateValueEstimates(self):
        sumy = [0.0, 0.0]
        for action in range(self.actionNum):
            if a == 1:
                nextState = (self.city+1)%2
            else:
                nextState = self.city
            futureEstimate = self.gamma * self.getValueForState(nextState)
            sumy[action] = self.R[self.city][action] + futureEstimate


        self.valueEstimate[self.city] = sumy

    def getValueForState(self, state):
        if self.valueEstimate[state][0] > self.valueEstimate[state][1]:
            return self.valueEstimate[state][0]
        else:
            return self.valueEstimate[state][1]

    def training(self):
        for i in range(1000):
            self.CalculateValueEstimates()
            action = self.getAction(self.city)
            if action == 1:
                self.city = (self.city+1) % 2
            # self.improveValueEstimate(self.city, action, reward)
            self.improvePolicy()
            print("Iteration {}: gave Estimates {}".format(i, self.valueEstimate))
        print(self.policy)


trainer = policyIteration(c, a, b, gamma, startCity, numberOfStates, numberOfActions)
trainer.training()