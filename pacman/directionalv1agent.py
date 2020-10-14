
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import networkx as nx 
import random
import bisect

def cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result
def choice(population, weights):
    cdf_vals = cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]

REVERSE_ACTION_MAPPING = {0:1, 1:0, 2:3, 3:2, 4:4} 
ACTION_TEXT_MAPPING = {0:'N', 1:'S', 2:'E', 3:'W', 4:'H'} 


class Directionalv1Agent():

    def __init__(self, index, posNodeMapping=None, nodeShortestDis=None, probAttack=0.5, probFlee=0.5):
        self.index = index
        self.posNodeMapping = posNodeMapping
        self.nodeShortestDis = nodeShortestDis
        self.probAttack = probAttack
        self.probFlee = probFlee

        self.speed = None
        self.isScared = None
        self.scareTimer = None

    def reset(self):
        self.speed = 1.0
        self.isScared = False
        self.scareTimer = None

    def getShortestDist(self, pacmanPos, ghostPosList):
        pacmanNode = self.posNodeMapping[pacmanPos]
        distList = []
        for ghostPos in ghostPosList:
            distList.append(self.nodeShortestDis[pacmanNode,self.posNodeMapping[ghostPos]])
        return distList

    def getAction(self, ghostLastAction, ghostActionList, ghostPosList, ghostPosIntList, pacmanPos):
        # Ghosts cannot stop, and cannot turn around unless they reach a dead end, but can turn 90 degrees at intersections.
        ghostRevLastAction = REVERSE_ACTION_MAPPING[ghostLastAction]
        if (ghostLastAction in ghostActionList) and (ghostRevLastAction in ghostActionList):
            idx = ghostActionList.index(ghostRevLastAction)
            del ghostActionList[idx]
            del ghostPosList[idx]
            del ghostPosIntList[idx]
        numAction = len(ghostActionList)
        distList = self.getShortestDist(pacmanPos,ghostPosIntList)
        if self.isScared:
            prob = [((1-self.probFlee)/numAction)] * numAction
            distMax = max(distList)
            candidates = [idx for idx,dist in enumerate(distList) if dist == distMax] 
            offset = self.probFlee/len(candidates)
        else:
            prob = [((1-self.probAttack)/numAction)] * numAction
            distMin = min(distList)
            candidates = [idx for idx,dist in enumerate(distList) if dist == distMin] 
            offset = self.probAttack/len(candidates)
        for candidate in candidates:
            prob[candidate] += offset
        idx = choice(range(numAction),prob) 
        return ghostActionList[idx],ghostPosList[idx]
