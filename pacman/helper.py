
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import os 
import numpy as np
import copy as cp
import networkx as nx
import torch

class Maze:
    def __init__(self, path):
        if (not os.path.exists(path)):
            raise Exception(path+' does not exist')
        else:
            # Read maze into numpy array
            self.name = os.path.splitext(os.path.basename(path))[0]
            with open(path) as f:
               self.maze = np.array([list(line.strip()) for line in f])
            # Pre-processing maze
            self.height, self.width = self.maze.shape
            
            self.wallRepr = (self.maze == '%')
            self.origStateRepr = np.zeros((5, self.height, self.width), dtype=bool)
            self.origStateRepr[0] = (self.maze == 'S')
            self.origStateRepr[1] = (self.maze == '.')
            self.origStateRepr[2] = (self.maze == 'o')
            self.origStateRepr[3] = (self.maze == 'P')
            self.origStateRepr[4] = (self.maze == 'G')
            
            self.numDots = int(np.sum(self.origStateRepr[1]))
            self.numFruits = int(np.sum(self.origStateRepr[2]))
            self.numGhosts = int(np.sum(self.origStateRepr[4]))
            self.wallPos = set(map(tuple,np.argwhere(self.wallRepr==True)))
            self.dotPos = set(map(tuple,np.argwhere(self.origStateRepr[1]==True)))
            self.fruitPos = set(map(tuple,np.argwhere(self.origStateRepr[2]==True)))
            self.origAgentPos = []
            self.origAgentPos.extend(map(tuple,np.argwhere(self.origStateRepr[3]==True).tolist()))
            self.origAgentPos.extend(map(tuple,np.argwhere(self.origStateRepr[4]==True).tolist()))
            self.origAgentDir = ['H']*(self.numGhosts+1)
            
            self.posNodeMapping = {}
            self.nodePosMapping = {}
            i = 0
            for row in range(self.wallRepr.shape[0]):
                for col in range(self.wallRepr.shape[1]):
                    if not self.wallRepr[row,col]:
                        self.posNodeMapping[(row,col)] = i
                        self.nodePosMapping[i] = (row,col)
                        i += 1
            graph = nx.Graph() 
            graph.add_nodes_from(range(i))
            for row in range(self.wallRepr.shape[0]):
                for col in range(self.wallRepr.shape[1]):
                    if (row,col) in self.posNodeMapping:
                        if (row-1,col) in self.posNodeMapping:
                            graph.add_edge(self.posNodeMapping[(row-1,col)], self.posNodeMapping[(row,col)])
                        if (row+1,col) in self.posNodeMapping:
                            graph.add_edge(self.posNodeMapping[(row+1,col)], self.posNodeMapping[(row,col)])
                        if (row,col-1) in self.posNodeMapping:
                            graph.add_edge(self.posNodeMapping[(row,col-1)], self.posNodeMapping[(row,col)])
                        if (row,col+1) in self.posNodeMapping:
                            graph.add_edge(self.posNodeMapping[(row,col+1)], self.posNodeMapping[(row,col)])
            self.nodeShortestDist = nx.floyd_warshall_numpy(graph).astype(int)
            self.maxShortestDist = np.max(self.nodeShortestDist)
        
    def getNumInputv1(self):
        return 5*self.height*self.width
    
ACTION_TEXT_MAPPING = {0:'N', 1:'S', 2:'E', 3:'W', 4:'H'} 
TEXT_ACTION_MAPPING = {'N':0, 'S':1, 'E':2, 'W':3, 'H':4} 
ACTION_VECTOR_MAPPING = {0:(-1,0), 1:(1,0), 2:(0,1), 3:(0,-1)}
ACTION_VECTOR_LIST = [(-1,0), (1,0), (0,1), (0,-1)]
TEXT_VECTOR_MAPPING = {'N':(-1,0), 'S':(1,0), 'E':(0,1), 'W':(0,-1)}
GRID_TOLERANCE = 0.001
SCARED_TIME = 40
COLLISION_TOLERANCE = 0.7

def dist(co1,co2):
    x1,y1 = co1
    x2,y2 = co2
    return abs(x1-x2)+abs(y1-y2)

def add(co1,co2):
    x1,y1 = co1
    x2,y2 = co2
    return (x1+x2,y1+y2)

def toInt(co1):
    x1,y1 = co1
    return (int(x1),int(y1))

def addInt(co1,co2):
    x1,y1 = co1
    x2,y2 = co2
    return (int(x1+x2),int(y1+y2))

def mul(f,co1):
    x1,x2 = co1
    return (f*x1,f*x2)

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Env:

    def __init__(self, maze, ghosts, maxStep):
        
        self.maze = maze
        self.posNodeMapping = self.maze.posNodeMapping
        self.nodeShortestDist = self.maze.nodeShortestDist
        self.maxShortestDist = self.maze.maxShortestDist
        self.mazeWidth = self.maze.width
        self.mazeHeight = self.maze.height
        self.ghosts = ghosts
        self.numGhosts = self.maze.numGhosts
        self.maxStep = maxStep

        self.observation_space = Namespace(shape=(5,self.mazeHeight,self.mazeWidth),n=5*self.mazeHeight*self.mazeWidth)
        self.action_space = Namespace(n=4)

        self.graphic = None
        self.wallRepr = self.maze.wallRepr
        self.wallPos = self.maze.wallPos
        self.origDotPos = self.maze.dotPos
        self.origFruitPos = self.maze.fruitPos
        self.origStateRepr = self.maze.origStateRepr
        self.stateRepr = None
        self.agentPos = None
        self.agentDir = None

        self.envStep = None
        self.gameOver = None
        self.curReward = None
        self.cumReward = None
        self.win = None
        self.eatenDotCoords = None
        self.eatenFruitCoords = None
        self.gameIndex = 0
    
    def setGraphic(self, graphic):
        self.graphic = graphic

    def reset(self, gameIndex=None):
        self.origAgentPos = self.maze.origAgentPos
        self.agentPos = cp.deepcopy(self.maze.origAgentPos)
        self.origAgentDir = self.maze.origAgentDir
        self.agentDir = cp.deepcopy(self.origAgentDir)
        self.dotPos = cp.deepcopy(self.origDotPos)
        self.fruitPos = cp.deepcopy(self.origFruitPos)
        self.eatenDotCoords = []
        self.eatenFruitCoords = []
        for ghost in self.ghosts:
            ghost.reset()

        self.gameOver = False
        self.win = False
        self.curReward = 0
        self.cumReward = 0
        self.envStep = 0

        if gameIndex != None:
            self.gameIndex = gameIndex 
        else:
            self.gameIndex += 1
        self.graphic.reset(ghosts=self.ghosts, wallRepr=self.wallRepr, stateRepr=self.origStateRepr, agentPos=self.agentPos, 
                agentDir=self.agentDir, eatenDotCoords=self.eatenDotCoords, eatenFruitCoords=self.eatenFruitCoords, gameIndex=self.gameIndex)
        return self.getStateReprv1()

    def getStateReprv1(self):
        stateRepr = torch.zeros(5, self.mazeHeight, self.mazeWidth)
        # Set Pacman position
        curPacmanPosIntX,curPacmanPosIntY = addInt(self.agentPos[0],(0.5,0.5))
        stateRepr[0,curPacmanPosIntX,curPacmanPosIntY] = 1.0
        # Set Ghost position
        for ghostIndex,ghost in enumerate(self.ghosts,start=1):
            curGhostPosIntX,curGhostPosIntY = addInt(self.agentPos[ghostIndex],(0.5,0.5))
            if ghost.isScared:
                stateRepr[1,curGhostPosIntX,curGhostPosIntY] = 1.0
            else:
                stateRepr[2,curGhostPosIntX,curGhostPosIntY] = 1.0
        # Set Dot position
        for row,col in self.dotPos:
            stateRepr[3,row,col] = 1.0
        # Set Fruit position
        for row,col in self.fruitPos:
            stateRepr[4,row,col] = 1.0
        
        return stateRepr

    def step(self, pacmanAction):

        self.curReward = 0
        
        # Move pacman
        curPacmanPosIntX,curPacmanPosIntY = addInt(self.agentPos[0],(0.5,0.5))
        newPacmanPosIntX,newPacmanPosIntY = newPacmanPosInt = addInt(add(self.agentPos[0],(0.5,0.5)),ACTION_VECTOR_MAPPING[pacmanAction])
        self.agentDir[0] = ACTION_TEXT_MAPPING[pacmanAction]
        if not newPacmanPosInt in self.wallPos:
            self.agentPos[0] = (newPacmanPosIntX, newPacmanPosIntY)
            # Collide with ghost
            for ghostIndex,ghost in enumerate(self.ghosts,start=1):
                if dist(newPacmanPosInt,self.agentPos[ghostIndex]) < COLLISION_TOLERANCE:
                    if ghost.isScared:
                        curGhostPosIntX,curGhostPosIntY = addInt(self.agentPos[ghostIndex],(0.5,0.5))
                        ghost.isScared = False
                        ghost.speed = 1.0
                        ghost.scareTimer = 0
                        origGhostPosX,origGhostPosY = self.origAgentPos[ghostIndex]
                        self.agentPos[ghostIndex] = self.origAgentPos[ghostIndex]
                        self.agentDir[ghostIndex] = self.origAgentDir[ghostIndex]
                        self.curReward += 1
                        self.cumReward += 1
                    else:
                        self.curReward -= 1
                        self.cumReward -= 1
                        self.gameOver = True
                        return self.getStateReprv1() ,self.curReward, self.gameOver, None
            # Collect dot and update dot location on state matrix
            if newPacmanPosInt in self.dotPos:
                self.dotPos.remove(newPacmanPosInt)
                self.eatenDotCoords.append((newPacmanPosIntX,newPacmanPosIntY))
                self.curReward += 1
                self.cumReward += 1
                if len(self.dotPos) == 0:
                    self.curReward += 1
                    self.cumReward += 1
                    self.gameOver = True
                    self.win = True
            # Collect dot and update dot location on state matrix
            if newPacmanPosInt in self.fruitPos:
                self.fruitPos.remove(newPacmanPosInt)
                self.eatenFruitCoords.append((newPacmanPosIntX,newPacmanPosIntY))
                self.curReward += 1
                self.cumReward += 1
                for ghostIndex,ghost in enumerate(self.ghosts,start=1):
                    curGhostPosInt = addInt(self.agentPos[ghostIndex],(0.5,0.5))
                    ghost.isScared = True
                    ghost.scareTimer = SCARED_TIME
                    ghost.speed = 0.5
        # self.graphic.update(self.cumReward)
        
        # Move ghosts
        for ghostIndex,ghost in enumerate(self.ghosts,start=1):
            curGhostPosIntX,curGhostPosIntY = curGhostPosInt = addInt(self.agentPos[ghostIndex],(0.5,0.5))
            # If ghost is between two grid, keep moving it in the same direction 
            if dist(self.agentPos[ghostIndex],curGhostPosInt) > GRID_TOLERANCE:
                if not ghost.isScared:
                    newGhostPos = add(self.agentPos[ghostIndex],mul(0.5,TEXT_VECTOR_MAPPING[self.agentDir[ghostIndex]]))
                else: 
                    newGhostPos = add(self.agentPos[ghostIndex],mul(ghost.speed,TEXT_VECTOR_MAPPING[self.agentDir[ghostIndex]]))
            # Otherwise, let ghost decide the best action
            else:
                ghostActionList = []
                ghostPosIntList = []
                ghostPosList = []
                for action,vector in enumerate(ACTION_VECTOR_LIST):
                    newGhostPosInt = addInt(add(self.agentPos[ghostIndex],(0.5,0.5)),vector)
                    if not newGhostPosInt in self.wallPos:
                        ghostActionList.append(action)
                        newGhostPos = add(self.agentPos[ghostIndex],mul(ghost.speed,vector))
                        ghostPosIntList.append(newGhostPosInt)
                        ghostPosList.append(newGhostPos)
                ghostAction,newGhostPos = ghost.getAction(ghostLastAction=TEXT_ACTION_MAPPING[self.agentDir[ghostIndex]], ghostActionList=ghostActionList, ghostPosList=ghostPosList, ghostPosIntList=ghostPosIntList, pacmanPos=self.agentPos[0])
                self.agentDir[ghostIndex] = ACTION_TEXT_MAPPING[ghostAction]
            self.agentPos[ghostIndex] = newGhostPos

            if ghost.isScared:
                ghost.scareTimer -= 1
                if ghost.scareTimer == 0:
                    ghost.isScared = False
                    ghost.speed = 1.0
                
            # Collide with pacman
            if dist(self.agentPos[0],newGhostPos) < COLLISION_TOLERANCE:
                if ghost.isScared:
                    ghost.isScared = False
                    ghost.speed = 1.0
                    ghost.scareTimer = 0
                    origGhostPosX,origGhostPosY = self.origAgentPos[ghostIndex]
                    self.agentPos[ghostIndex] = self.origAgentPos[ghostIndex]
                    self.agentDir[ghostIndex] = self.origAgentDir[ghostIndex]
                    self.curReward += 1
                    self.cumReward += 1
                else:
                    self.curReward -= 1
                    self.cumReward -= 1
                    self.gameOver = True
                    return self.getStateReprv1() ,self.curReward, self.gameOver, None
            # self.graphic.update(self.cumReward)
        self.graphic.update(self.cumReward)
           
        # Conclude episode if maximum number of episodes is reached
        self.envStep += 1
        if self.envStep > self.maxStep:
            self.gameOver = True
        return self.getStateReprv1() ,float(self.curReward), self.gameOver, None
