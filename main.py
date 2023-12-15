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
import math
import time
import copy

class Obstacle():
    def __init__(self, points, screen):
        self.points = points
        self.surface = screen 
    
    def draw(self):
        pygame.draw.polygon(self.surface, "red", self.points)
        pygame.display.flip()

    def getPoints(self):
        return self.points

class Robot():
    def __init__(self, center, points, screen):
        self.center = center
        self.points = points
        self.surface = screen

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
    
    def printLocation(self):
        print("center: " + str(self.center))
        print("points: " + str(self.points))

    def getCenter(self):
        return self.center

    def getPoints(self):
        return self.points

class Simulator():
    def __init__(self, screen, w, l):
        self.screen = screen
        self.start = [40, 100, 0]
        self.end = [600, 400, 45]
        self.w = w
        self.l = l
        self.robot = Robot(self.start[0:2], \
            [[20, 50], [20, 150], [60, 150], [60, 50]], screen)
        self.obstacles = [
            Obstacle([[300, 100], [300, 400], [400, 100]], screen),
            Obstacle([[0, 300], [200, 300], [200, 375], [0, 375]], screen),
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
                    self.traversePath()
                    # resets the screen

            self.draw_all()
                    
    def draw_all(self):
        for obs in self.obstacles:
            obs.draw()
        pygame.draw.circle(self.screen, "blue", self.start[0:2], 3)
        pygame.draw.circle(self.screen, "blue", self.end[0:2], 3)
        self.robot.draw()
    
    def create_graph(self, shift, angle):

        def inside_convex_polygon(p, v):
            prev_side = None
            for i in range(len(v)):
                a, b = v[i], v[(i+1)%len(v)]
                segment_to_point = (p[0] - a[0], p[1] - a[1])
                segment_of_vertices = (b[0] - a[0], b[1] - a[1])
                det = segment_of_vertices[0] * segment_to_point[1] - \
                    segment_of_vertices[1] * segment_to_point[0]
                if(det == 0):
                    # point on line 
                    return False
                elif(det > 0):
                    curr_side = "R"
                else:
                    curr_side = "L"
                if(prev_side == None):
                    prev_side = curr_side
                elif(prev_side != curr_side):
                    return False

            return True
        
        def isValid(p):
            # checks if a point is in the obstacles
            for obs in self.obstacles:
                if(inside_convex_polygon(p, obs.getPoints())):
                    return False

            return True

        robot_pts = self.robot.getPoints().copy()
        robot_center = self.robot.getCenter().copy()
        robot = Robot(robot_center, robot_pts, self.screen)
        robot.move(-(robot.getCenter()[0]), -(robot.getCenter()[1]))

        a = 0 # current angle
        pos_id = 1 # an id for each cell of the matrix
        cols = math.ceil(self.l / shift)
        rows = math.ceil(self.w / shift)
        depth = 360 // angle
        adjacencyMatrix = []

        for k in range(depth):
            adjacencyMatrix.append([])

            # for each angle ... 
            robot.rotate(a)
            for c in range(cols):
                adjacencyMatrix[k].append([])
                for r in range(rows):
                    print(robot.getCenter())
                    if isValid(robot.getCenter()):
                        adjacencyMatrix[k][c].append([r * shift, c * shift, a, False])
                        pos_id += 1
                    else:
                        adjacencyMatrix[k][c].append([r * shift, c * shift, a, True])
                    robot.move(shift, 0)
                robot.move(-(shift * rows), shift)
            robot.move(0, -(shift * cols))
            a+=angle

        self.adjacencyMatrix = adjacencyMatrix
        self.path = self.bfsPathFind(shift, angle)

        # print(adjacencyMatrix)

    def bfsPathFind(self, shift, angle):
        adjacencyMatrix = self.adjacencyMatrix.copy()

        queue = []
        s = self.start.copy()
        queue.append([[s[2]//angle, s[1]//shift, s[0]//shift]])
        adjacencyMatrix[s[2]//angle][s[1]//shift][s[0]//shift][3] = True
        dirs = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]

        while(queue):
            path = queue.pop(0)
            x_i, y_i, z_i = path[-1]
            node_val = adjacencyMatrix[x_i][y_i][z_i]
            x, y, z, v = node_val
            if(node_val[0:3] == self.end):
                return path
            for d in dirs:
                a = d[0]
                b = d[1]
                c = d[2]
                if(z + a * angle >= 0 and z + a * angle < 360
                and y + b * shift >= 0 and y + b * shift < self.l
                and x + c * shift >= 0 and x + c * shift < self.w and not
                adjacencyMatrix[x_i+a][y_i+b][z_i+c][3]):
                    adjacencyMatrix[x_i+a][y_i+b][z_i+c][3] = True
                    newpath = path.copy()
                    newpath.append([x_i+a,y_i+b,z_i+c])
                    queue.append(newpath)

        return -1

    def traversePath(self):
        if(self.path == -1):
            print("No path found")
            return
        x_p, y_p, a_p = self.start
        for x_i, y_i, z_i in self.path:
            time.sleep(0.1)
            x, y, a, v = self.adjacencyMatrix[x_i][y_i][z_i]
            print(self.adjacencyMatrix[x_i][y_i][z_i])
            print(x - x_p, y - y_p, a-a_p)
            self.robot.move(x - x_p, y - y_p)
            self.robot.rotate(a - a_p)
            self.screen.fill((255, 255, 255))
            self.draw_all()
            x_p = x
            y_p = y
            a_p = a


def main():
    pygame.init()
    pygame.display.set_caption("Robot Simulation")
    screen = pygame.display.set_mode((640, 480))
    simulation = Simulator(screen, 640, 480)
    simulation.create_graph(10, 45)
    simulation.main()
main()