"""
Nicholas Fettig
CPSC 091T: Computational Geometry
Prof. Neil Lutz
Fall 2023

Final Project:
Rigorous Movement of Convex Polygons on a Path Using Multiple Robots Simulation

DESCRIPTION

pip libs:
pygame

file deps:
*.py
"""

import pygame
import numpy as np
import time
import copy

class Obstacle():
    def __init__(self, points, screen):
        self.points = points
        self.surface = screen 
    
    def draw(self):
        pygame.draw.polygon(self.surface, "red", self.points)
        pygame.display.flip()

class Robot():
    def __init__(self, center, points, screen):
        self.center = center
        self.points = points
        self.surface = screen
    
    def setX(self, x):
        self.x = x
    
    def setY(self, y):
        self.y = y

    def setAngle(self, angle):
        self.orientation = angle

    def move(self, x, y):
        curr_x, curr_y = self.center 
        self.center = [curr_x + x, curr_y + y]
        for i in range(len(self.points)):
            curr_x, curr_y = self.points[i]
            self.points[i] = [curr_x + x, curr_y + y]

    def rotate(self, angle):
        ang_r = angle * np.pi / 180
        x_c, y_c = self.center
        rotated_pts = []
        for pt in self.points:
            x_new = x_c + np.cos(ang_r) * (pt[0] - x_c) - np.sin(ang_r) * (pt[1] - y_c)
            y_new = y_c + np.sin(ang_r) * (pt[0] - x_c) + np.cos(ang_r) * (pt[1] - y_c)
            rotated_pts.append([x_new, y_new])
        self.points = rotated_pts

    def draw(self):
        pygame.draw.polygon(self.surface, "black", self.points)
        pygame.draw.circle(self.surface, "red", self.center, 3)
        pygame.display.flip()

class Simulator():
    def __init__(self, screen, w, l):
        self.screen = screen
        self.start = [0, 0]
        self.end = [300, 500]
        self.w = w
        self.l = l
        self.robot = Robot(self.start, \
            [[10, 50], [10, 150], [60, 150], [60, 50]], screen)
        self.obstacles = [
            Obstacle([[300, 100], [300, 400], [400, 100]], screen),
            Obstacle([[50, 300], [200, 300], [200, 375], [50, 375]], screen),
            Obstacle([[600, 200], [600, 300], [450, 300]], screen)
        ]
        self.adjacencyMatrix = []

    def main(self): 
        clock = pygame.time.Clock()
        self.screen.fill((255, 255, 255))
        self.draw_all()
        while 1:
            clock.tick(100)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                elif event.type == pygame.KEYDOWN:
                    self.robot.move(10, 10)
                    self.robot.rotate(10)
                    # resets the screen
                    self.screen.fill((255, 255, 255))

            self.draw_all()
                    
    def draw_all(self):
        for obs in self.obstacles:
            obs.draw()
        self.robot.draw()
        pygame.draw.circle(self.screen, "blue", self.start, 3)
        pygame.draw.circle(self.screen, "blue", self.end, 3)
    
    def create_graph(self, shift, angle):

        a = 0 # current angle
        pos_id = 1 # an id for each cell of the matrix
        cols = self.w // shift
        rows = self.l // shift
        depth = 360 // angle
        adjacencyMatrix = []

        for k in range(depth):
            adjacencyMatrix.append([])
            # for each angle ... 
            self.robot.rotate(a)
            for c in range(cols):
                adjacencyMatrix[k].append([])
                for r in range(rows):
                    self.robot.move(shift, 0)
                    if 1:
                        adjacencyMatrix[k][c].append([r * shift, c * shift, a, False])
                        pos_id += 1
                    else:
                        adjacencyMatrix[k][c].append(0)
                self.robot.setX(0)
                self.robot.move(0, shift)
            a+=angle

        self.adjacencyMatrix = adjacencyMatrix
        path = self.bfsPathFind(shift, angle)
        print(path)

    def bfsPathFind(self, shift, angle):
        adjacencyMatrix = self.adjacencyMatrix
        queue = []
        queue.append([[0, 0, 0]])
        adjacencyMatrix[0][0][0][3] = True
        while(queue):
            path = queue.pop(0)
            x_i, y_i, z_i = path[-1]
            node_val = adjacencyMatrix[x_i][y_i][z_i]
            x, y, z, v = node_val
            print(x, y, z, v)
            if(x == self.end[0] and y == self.end[1]):
                return path
            for a in [-1, 1]:
                for b in [-1, 1]:
                    for c in [-1, 1]:
                        if(x + a * angle >= 0 and x + a * angle < 360
                        and y + b * shift >= 0 and y + b * shift < self.w
                        and z + c * shift >= 0 and z + c * shift < self.l and not
                        adjacencyMatrix[x_i+a][y_i+b][z_i+c][3]):
                            adjacencyMatrix[x_i+a][y_i+b][z_i+c][3] = True
                            newpath = path
                            newpath.append([x_i+a,y_i+b,z_i+c])
                            print(newpath)
                            queue.append(newpath)

        return -1

def main():
    pygame.init()
    pygame.display.set_caption("Robot Simulation")
    screen = pygame.display.set_mode((640, 480))
    simulation = Simulator(screen, 640, 480)
    simulation.create_graph(100, 45)
    simulation.main()
main()