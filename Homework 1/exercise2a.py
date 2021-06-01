C = -3
rA = 2
rB = 1
gamma = 0.9

numTimes = 100
a = [(gamma**x * rA) for x in range(numTimes)]
print(sum(a))

b = [(gamma**y * rB) for y in range(numTimes)]
print(sum(b))

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
