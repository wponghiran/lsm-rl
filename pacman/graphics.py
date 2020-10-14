
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import tkinter as tk
import os
import numpy as np
import pathlib

DEFAULT_GRID_SIZE = 30.0
INFO_PANE_HEIGHT = 35
BACKGROUND_COLOR = '#000000'
WALL_COLOR = '#0033ff'
INFO_PANE_COLOR ='#666600'
SCOREBOARD_COLOR ='#ffff3d'
PACMAN_OUTLINE_WIDTH = 2
PACMAN_CAPTURE_OUTLINE_WIDTH = 4

GHOST_COLORS = ['#e50000', '#004ce5', '#f96811', '#19bfb2', '#ff9900', '#6621e8']

GHOST_SHAPE = np.array([
    (0,    0.3),
    (0.25, 0.75),
    (0.5,  0.3),
    (0.75, 0.75),
    (0.75, -0.5),
    (0.5,  -0.75),
    (-0.5,  -0.75),
    (-0.75, -0.5),
    (-0.75, 0.75),
    (-0.5,  0.3),
    (-0.25, 0.75)
])
GHOST_SIZE = 0.6
SCARED_COLOR = '#ffffff'
GHOST_EYE_COLOR = '#ffffff'
GHOST_PUPIL_COLOR = '#000000'

PACMAN_COLOR ='#ffff3d'
PACMAN_SIZE = 0.4
PACMAN_DELTA = 30

# Food
FOOD_COLOR = '#ffffff'
FOOD_SIZE = 0.1

# Capsule graphics
CAPSULE_COLOR = '#ffffff'
CAPSULE_SIZE = 0.25

# Drawing walls
WALL_RADIUS = 0.15


class Graphics:
    def __init__(self, maze, save=False):
        self.save = save
        self.have_window = 0
        self.currentGhostImages = {}
        self.pacmanImage = None
        self.gridSize = DEFAULT_GRID_SIZE
        self.wallRadius = WALL_RADIUS*self.gridSize
        self.pacmanSize = PACMAN_SIZE*self.gridSize
        self.ghostSize = GHOST_SIZE*self.gridSize
        self.ghostShape = GHOST_SHAPE*self.ghostSize 
        self.foodSize = FOOD_SIZE*self.gridSize
        self.capsuleSize = CAPSULE_SIZE*self.gridSize
       
        self.mazeHeight = (maze.height+0.5)*self.gridSize
        self.width = (maze.width+1)*self.gridSize
        self.height = (maze.height+1)*self.gridSize+INFO_PANE_HEIGHT

        self.window = None      # The root window for graphics output
        self.canvas = None      # The canvas which holds graphics

        self.gameIndex = None
        self.frameIndex = None
        self.savePath = 'frames'
        pathlib.Path(os.path.abspath(self.savePath)).mkdir(parents=True, exist_ok=True)
        self.savePrefix = maze.name  

    def saveFrame(self):
        with open(os.path.join(self.savePath, '%s_%03d_%08d.ps' % (self.savePrefix, self.gameIndex, self.frameIndex)), 'w') as f:
            f.write(self.canvas.postscript(pageanchor='sw', y='0.c', x='0.c'))
        self.frameIndex += 1

    def drawPolygon(self, coords, outlineColor='', fillColor='', smooth=True, behind=0, width=1):
        polygon = self.canvas.create_polygon([coord for pair in coords for coord in pair],  outline=outlineColor, fill=fillColor, smooth=smooth, width=width)
        if behind > 0:
            self.canvas.tag_lower(poly, behind)  # Higher should be more visible
        return polygon

    def drawCircle(self, pos, r, outlineColor='', fillColor='', startEndPoints=None, style='pieslice', width=2):
        x, y = pos
        x0, x1 = x-r-1,x+r
        y0, y1 = y-r-1,y+r
        if startEndPoints == None:
            startEndPoints = (0, 359)
        return self.canvas.create_arc(x0, y0, x1, y1, outline=outlineColor, fill=fillColor, start=startEndPoints[0], extent=startEndPoints[1]-startEndPoints[0], style=style, width=width)

    def drawLine(self, here, there, color='#000000', width=2):
        x0, y0 = here[0], here[1]
        x1, y1 = there[0], there[1]
        return self.canvas.create_line(x0, y0, x1, y1, fill=color, width=width)
    
    def scale(self, coord):
        return (np.array(coord)+1)*self.gridSize

    def refresh(self):
        self.canvas.update_idletasks()
    
    def createWindow(self, color=BACKGROUND_COLOR, title=None):
    
        # Close existing window before opening a new one
        if not (self.window is None):
            self.window.destroy()
    
        # Create the root window
        self.window = tk.Tk()
        self.window.geometry('+%d+%d'%(20,75))  
        self.window.title(title or 'Graphics Window')
        self.window.resizable(0, 0)
    
        # Create the canvas object
        self.canvas = tk.Canvas(self.window, width=self.width+1, height=self.height+1)
        self.canvas.pack()
        corners = [(0,0), (0,self.height), (self.width,self.height), (self.width,0)]
        self.drawPolygon(corners, outlineColor=color, fillColor=color, smooth=False)
        self.canvas.update()

    # This function is called at the begining of the game to initialize graphic
    def reset(self, ghosts, wallRepr, stateRepr, agentPos, agentDir, eatenDotCoords, eatenFruitCoords, gameIndex=0):
        self.ghosts = ghosts
        self.numGhosts = len(self.ghosts)
        self.agentPos = agentPos
        self.agentDir = agentDir
        self.eatenDotCoords = eatenDotCoords 
        self.eatenFruitCoords = eatenFruitCoords 
        self.gameIndex = gameIndex
        self.frameIndex = 0
        self.createWindow(title="Pacman Game")

        self.scoreBoard = self.canvas.create_text(self.gridSize, self.mazeHeight, fill=SCOREBOARD_COLOR, text='SCORE:    0', font=('Times', '24', 'bold'), anchor='nw')
        self.drawWall(wallRepr)
        self.dotObj = self.drawFood(stateRepr[1])
        self.fruitObj = self.drawFruit(stateRepr[2])
        self.pacmanCoord,self.pacmanObj = self.drawPacman(self.agentPos[0],self.agentDir[0])
        self.ghostCoord,self.ghostObj = self.drawGhost(self.agentPos[1:],self.agentDir[1:])
        self.refresh()
        # time.sleep(0.1)
        if self.save:  self.saveFrame()

    def update(self, accReward):
        self.canvas.itemconfigure(self.scoreBoard, text=('SCORE: %4d'%(accReward)))
        while self.eatenDotCoords:
            coord = self.eatenDotCoords.pop()
            self.canvas.delete(self.dotObj[coord])
        while self.eatenFruitCoords:
            coord = self.eatenFruitCoords.pop()
            self.canvas.delete(self.fruitObj[coord])
        self.movePacman(self.agentPos[0],self.agentDir[0])
        self.moveGhost(self.agentPos[1:],self.agentDir[1:],[ghost.isScared for ghost in self.ghosts])
        self.refresh()
        # time.sleep(0.1)
        if self.save:  self.saveFrame()

    # def drawPacman(self, (row,col), direction):
    def drawPacman(self, co, direction):
        row,col = co
        screen = self.scale((col,row))
        if (direction == 'W'):  startEndPoints = (180+PACMAN_DELTA, 540-PACMAN_DELTA)
        elif (direction == 'N'):    startEndPoints = (90+PACMAN_DELTA, 450-PACMAN_DELTA)
        elif (direction == 'S'):    startEndPoints = (270+PACMAN_DELTA, 630-PACMAN_DELTA)
        elif (direction == 'E'):    startEndPoints = (0+PACMAN_DELTA, 360-PACMAN_DELTA)
        else:    startEndPoints = (0+PACMAN_DELTA, 360-PACMAN_DELTA)
        return (screen,self.drawCircle(screen, self.pacmanSize, fillColor=PACMAN_COLOR, outlineColor=PACMAN_COLOR, startEndPoints=startEndPoints, width=PACMAN_OUTLINE_WIDTH))
   
    def moveObj(self, obj, deltaScreen):
        newCoords = []
        isCol = True
        for coord in self.canvas.coords(obj):
            if isCol:   newCoords.append(coord+deltaScreen[0])
            else:   newCoords.append(coord+deltaScreen[1])
            isCol = not isCol
        # Move pacman to new postion
        self.canvas.coords(obj, *newCoords)
    
    # def movePacman(self, (newRow,newCol), newDir):
    def movePacman(self, loc, newDir):
        newRow,newCol = loc
        # Calculate difference in position and new coordinate
        deltaScreen = self.scale((newCol,newRow))-self.pacmanCoord
        self.pacmanCoord = self.scale((newCol,newRow))
        self.moveObj(self.pacmanObj, deltaScreen)
        # Edit position of pacman's mouth
        if (newDir == 'W'):  startEndPoints = (180+PACMAN_DELTA, 540-PACMAN_DELTA)
        elif (newDir == 'N'):    startEndPoints = (90+PACMAN_DELTA, 450-PACMAN_DELTA)
        elif (newDir == 'S'):    startEndPoints = (270+PACMAN_DELTA, 630-PACMAN_DELTA)
        elif (newDir == 'E'):   startEndPoints = (0+PACMAN_DELTA, 360-PACMAN_DELTA)
        else:    startEndPoints = (0+PACMAN_DELTA, 360-PACMAN_DELTA)
        self.canvas.itemconfigure(self.pacmanObj, start=startEndPoints[0], extent=startEndPoints[1]-startEndPoints[0]) 

    def drawGhost(self, ghostPos, ghostDir):
        ghostCoord = {}
        ghostObj = {}
        for ghostIndex in range(self.numGhosts):
            (row,col) = ghostPos[ghostIndex]
            direction = ghostDir[ghostIndex]
            screen = self.scale((col,row))
            coords = np.copy(self.ghostShape)+screen
            body = self.drawPolygon(coords, outlineColor=GHOST_COLORS[ghostIndex], fillColor=GHOST_COLORS[ghostIndex])
            dx = 0
            dy = 0
            if direction == 'W': dx = -0.2
            elif direction == 'N': dy = -0.2
            elif direction == 'S': dy = 0.2
            elif direction == 'E': dx = 0.2
            lEye = self.drawCircle(screen+(self.ghostSize*(-0.3+dx/1.5),-self.ghostSize*(0.3-dy/1.5)), self.ghostSize*0.2, GHOST_EYE_COLOR, GHOST_EYE_COLOR)
            rEye = self.drawCircle(screen+(self.ghostSize*( 0.3+dx/1.5),-self.ghostSize*(0.3-dy/1.5)), self.ghostSize*0.2, GHOST_EYE_COLOR, GHOST_EYE_COLOR)
            lPupil = self.drawCircle(screen+(self.ghostSize*(-0.3+dx),-self.ghostSize*(0.3-dy)), self.ghostSize*0.08, GHOST_PUPIL_COLOR, GHOST_PUPIL_COLOR)
            rPupil = self.drawCircle(screen+(self.ghostSize*( 0.3+dx),-self.ghostSize*(0.3-dy)), self.ghostSize*0.08, GHOST_PUPIL_COLOR, GHOST_PUPIL_COLOR)
            ghostCoord[ghostIndex] = screen
            ghostObj[ghostIndex] = [body, rEye, lEye, lPupil, rPupil]

        return (ghostCoord,ghostObj)

    def moveGhost(self, newGhostPos, newGhostDir, isScared):
        for ghostIndex in range(self.numGhosts):
            (newRow,newCol) = newGhostPos[ghostIndex]
            # Calculate difference in position and new coordinate
            deltaScreen = self.scale((newCol,newRow))-self.ghostCoord[ghostIndex]
            self.ghostCoord[ghostIndex] = self.scale((newCol,newRow)) 
            for obj in self.ghostObj[ghostIndex]:
                self.moveObj(obj, deltaScreen)
            # Set color of ghost
            if isScared[ghostIndex]:
                self.canvas.itemconfigure(self.ghostObj[ghostIndex][0], fill=SCARED_COLOR, outline=SCARED_COLOR) 
            else:
                self.canvas.itemconfigure(self.ghostObj[ghostIndex][0], fill=GHOST_COLORS[ghostIndex], outline=GHOST_COLORS[ghostIndex]) 

    def drawWall(self, wallRepr):
        wallColor = WALL_COLOR
        for row in range(wallRepr.shape[0]):
            for col in range(wallRepr.shape[1]):
                if wallRepr[row, col]:

                    # Draw each quadrant of the square based on adjacent walls
                    wIsWall = self.isWall(row, col-1, wallRepr)
                    eIsWall = self.isWall(row, col+1, wallRepr)
                    nIsWall = self.isWall(row-1, col, wallRepr)
                    sIsWall = self.isWall(row+1, col, wallRepr)
                    nwIsWall = self.isWall(row-1, col-1, wallRepr)
                    swIsWall = self.isWall(row+1, col-1, wallRepr)
                    neIsWall = self.isWall(row-1, col+1, wallRepr)
                    seIsWall = self.isWall(row+1, col+1, wallRepr)

                    # Original pacman source code arrages position as (col,row)
                    # We keep at the same order
                    screen = self.scale((col, row))

                    # NE quadrant
                    if (not nIsWall) and (not eIsWall):
                        # inner circle
                        self.drawCircle(screen, self.wallRadius, wallColor, wallColor, (0, 91), 'arc')
                    if (nIsWall) and (not eIsWall):
                        # vertical line
                        self.drawLine(add(screen, (self.wallRadius, 0)), add(screen, (self.wallRadius, self.gridSize * (-0.5) - 1)), wallColor)
                    if (not nIsWall) and (eIsWall):
                        # horizontal line
                        self.drawLine(add(screen, (0, -1*self.wallRadius)), add(screen, (self.gridSize * 0.5 + 1, -1*self.wallRadius)), wallColor)
                    if (nIsWall) and (eIsWall) and (not neIsWall):
                        # outer circle
                        self.drawCircle(add(screen, (2*self.wallRadius, -2*self.wallRadius)), self.wallRadius - 1, wallColor, wallColor, (180, 271), 'arc')
                        self.drawLine(add(screen, (2*self.wallRadius - 1, -1*self.wallRadius)), add(screen, (self.gridSize * 0.5 + 1, -1*self.wallRadius)), wallColor)
                        self.drawLine(add(screen, (self.wallRadius, -2*self.wallRadius + 1)), add(screen, (self.wallRadius, self.gridSize * (-0.5))), wallColor)

                    # NW quadrant
                    if (not nIsWall) and (not wIsWall):
                        # inner circle
                        self.drawCircle(screen, self.wallRadius, wallColor, wallColor, (90, 181), 'arc')
                    if (nIsWall) and (not wIsWall):
                        # vertical line
                        self.drawLine(add(screen, (-1*self.wallRadius, 0)), add(screen, (-1*self.wallRadius, self.gridSize * (-0.5) - 1)), wallColor)
                    if (not nIsWall) and (wIsWall):
                        # horizontal line
                        self.drawLine(add(screen, (0, -1*self.wallRadius)), add(screen, (self.gridSize * (-0.5) - 1, -1*self.wallRadius)), wallColor)
                    if (nIsWall) and (wIsWall) and (not nwIsWall):
                        # outer circle
                        self.drawCircle(add(screen, (-2*self.wallRadius, -2*self.wallRadius)), self.wallRadius - 1, wallColor, wallColor, (270, 361), 'arc')
                        self.drawLine(add(screen, (-2*self.wallRadius + 1, -1*self.wallRadius)), add(screen, (self.gridSize * (-0.5), -1*self.wallRadius)), wallColor)
                        self.drawLine(add(screen, (-1*self.wallRadius, -2*self.wallRadius + 1)), add(screen, (-1*self.wallRadius, self.gridSize * (-0.5))), wallColor)

                    # SE quadrant
                    if (not sIsWall) and (not eIsWall):
                        # inner circle
                        self.drawCircle(screen, self.wallRadius, wallColor, wallColor, (270, 361), 'arc')
                    if (sIsWall) and (not eIsWall):
                        # vertical line
                        self.drawLine(add(screen, (self.wallRadius, 0)), add(screen,(self.wallRadius, self.gridSize * (0.5) + 1)), wallColor)
                    if (not sIsWall) and (eIsWall):
                        # horizontal line
                        self.drawLine(add(screen, (0, self.wallRadius)), add(screen,(self.gridSize * 0.5 + 1, self.wallRadius)), wallColor)
                    if (sIsWall) and (eIsWall) and (not seIsWall):
                        # outer circle
                        self.drawCircle(add(screen, (2*self.wallRadius, 2*self.wallRadius)), self.wallRadius - 1, wallColor, wallColor, (90, 181), 'arc')
                        self.drawLine(add(screen, (2*self.wallRadius - 1, self.wallRadius)), add(screen, (self.gridSize * 0.5, self.wallRadius)), wallColor)
                        self.drawLine(add(screen, (self.wallRadius, 2*self.wallRadius - 1)), add(screen, (self.wallRadius, self.gridSize * (0.5))), wallColor)

                    # SW quadrant
                    if (not sIsWall) and (not wIsWall):
                        # inner circle
                        self.drawCircle(screen, self.wallRadius, wallColor, wallColor, (180, 271), 'arc')
                    if (sIsWall) and (not wIsWall):
                        # vertical line
                        self.drawLine(add(screen, (-1*self.wallRadius, 0)), add(screen,(-1*self.wallRadius, self.gridSize * (0.5) + 1)), wallColor)
                    if (not sIsWall) and (wIsWall):
                        # horizontal line
                        self.drawLine(add(screen, (0, self.wallRadius)), add(screen,(self.gridSize * (-0.5) - 1, self.wallRadius)), wallColor)
                    if (sIsWall) and (wIsWall) and (not swIsWall):
                        # outer circle
                        self.drawCircle(add(screen, (-2*self.wallRadius, 2*self.wallRadius)), self.wallRadius - 1, wallColor, wallColor, (0, 91), 'arc')
                        self.drawLine(add(screen, (-2*self.wallRadius + 1, self.wallRadius)), add(screen, (self.gridSize * (-0.5), self.wallRadius)), wallColor)
                        self.drawLine(add(screen, (-1*self.wallRadius, 2*self.wallRadius - 1)), add(screen, (-1*self.wallRadius, self.gridSize * (0.5))), wallColor)

    def isWall(self, x, y, wallRepr):
        if x < 0 or y < 0 or x >= wallRepr.shape[0] or y >= wallRepr.shape[1]:
            return False
        else:
            return wallRepr[x,y]

    def drawFood(self, dotRepr):
        dotObj = {}
        for row in range(dotRepr.shape[0]):
            for col in range(dotRepr.shape[1]):
                if dotRepr[row,col]:
                    screen = self.scale((col,row))
                    dotObj[(row,col)] = self.drawCircle(screen, self.foodSize, outlineColor=FOOD_COLOR, fillColor=FOOD_COLOR, width=1)
        return dotObj

    def drawFruit(self, fruitRepr):
        fruitObj = {}
        for row in range(fruitRepr.shape[0]):
            for col in range(fruitRepr.shape[1]):
                if fruitRepr[row,col]:
                    screen = self.scale((col,row))
                    fruitObj[(row,col)] = self.drawCircle(screen, self.capsuleSize, outlineColor=CAPSULE_COLOR, fillColor=CAPSULE_COLOR, width=1)
        return fruitObj

class NullGraphics:
    def __init__(self):
        pass
    
    def reset(self, ghosts, wallRepr, stateRepr, agentPos, agentDir, eatenDotCoords, eatenFruitCoords, gameIndex=0):
        pass
    
    def update(self, accReward):
        pass

def add(x,y):
    return x+y    
