
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import sys
import time
import argparse

from . import helper 
from . import graphics
from . import directionalv1agent

def init_env(maze_path='./pacman/mazes/test.maze', save_frame=False, display=False):
    setting = {}

    # Initialize maze
    setting['maze'] = helper.Maze(maze_path)
    
    # Initialize ghost
    Ghost = directionalv1agent.Directionalv1Agent
    setting['ghosts'] = [Ghost(i+1, posNodeMapping=setting['maze'].posNodeMapping, nodeShortestDis=setting['maze'].nodeShortestDist) for i in range(setting['maze'].numGhosts)]

    # Load graphic
    setting['maxStep'] = 10000
    setting['save_frame'] = save_frame
    
    # Initialize environment
    env = helper.Env(maze=setting['maze'], ghosts=setting['ghosts'], maxStep=setting['maxStep'])

    if not display:
        graphic = graphics.NullGraphics()
    else:
        graphic = graphics.Graphics(maze=setting['maze'], save=setting['save_frame'])
    env.setGraphic(graphic) # Change graphic for replay
    
    return env

def loadAgent(agentName):
    try:
        module = __import__('pacman.'+agentName.lower())
        return getattr(module, agentName)
    except ImportError:
        raise Exception('Fail to import the agent ' + agentName+'.')
         
