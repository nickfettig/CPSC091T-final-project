"""
Nicholas Fettig
CPSC 091T: Computational Geometry
Prof. Neil Lutz
Fall 2023

Final Project:
Rigorous Movement of Convex Polygons on a Path Using Multiple Robots Simulation

DESCRIPTION

pip libs:
pygame, numpy

"""

import pygame
import numpy as np
import math
import time

class Obstacle():
    """ Defines the obstacle positions """

    def __init__(self, points, screen):
        self.points = points
        self.surface = screen 
    
    def draw(self):
        pygame.draw.polygon(self.surface, "red", self.points)
        pygame.display.flip()

    def getPoints(self):
        return self.points

class Robot():
    """ 
        Handles robot positioning, movement, and rotation
        Note that all operations are local and in-place
    """
    def __init__(self, center, points, screen):
        self.center = center
        self.points = points
        self.surface = screen
        self.initial_center = center
        self.inital_points = points

    def move(self, x, y):
        """ Will move the robot a set x and y from its current location """
        curr_x, curr_y = self.center 
        self.center = [curr_x + x, curr_y + y]
        for i in range(len(self.points)):
            curr_x, curr_y = self.points[i]
            self.points[i] = [curr_x + x, curr_y + y]

    def rotate(self, angle):
        """ Rotates the robot by an angle by reference of its center point"""
        ang_r = angle * np.pi / 180
        x_c, y_c = self.center
        rotated_pts = []
        for pt in self.points:
            x_new = x_c + np.cos(ang_r) * (pt[0] - x_c) - np.sin(ang_r) * (pt[1] - y_c)
            y_new = y_c + np.sin(ang_r) * (pt[0] - x_c) + np.cos(ang_r) * (pt[1] - y_c)
            rotated_pts.append([x_new, y_new])
        self.points = rotated_pts

    def draw(self):
        """ Draws robot to screen """
        pygame.draw.polygon(self.surface, "black", self.points)
        pygame.draw.circle(self.surface, "red", self.center, 3)
        pygame.display.flip()
    
    def printLocation(self):
        """ Prints robot information """
        print("center: " + str(self.center))
        print("points: " + str(self.points))

    def getCenter(self):
        """ returns center """
        return self.center

    def getPoints(self):
        """ returns list of points """
        return self.points
    
    def reset(self):
        self.points = self.inital_points
        self.center = self.initial_center

class Simulator():
    """ 
    Handles operation of the simulation. Controlls robot / obstacles 
    and holds all necessary algorithms.
    """
    def __init__(self, screen, w, l):
        self.screen = screen
        self.start = [40, 100, 0]
        self.end = [560, 400, 0]
        self.w = w
        self.l = l

        """ 
        Play around with these values to test different robots through courses!

        NOTE: ALL POINTS MUST BE IN CCW ORDER FOR MINKOWSKI SUM TO WORK
        """
        self.robot = Robot(self.start[0:2], \
            [[20, 50], [20, 150], [60, 150], [80, 100], [60, 50]], screen)
        self.obstacles = [
            Obstacle([[300, 100], [300, 400], [400, 100]], screen),
            Obstacle([[0, 300], [0, 375], [200, 375], [200, 300]], screen),
            Obstacle([[450, 300], [600, 300], [600, 200]], screen)
        ]

        """ Used for path-finding """
        self.adjacencyMatrix = []
        self.path = []
        self.shift = 10
        self.angle = 5

    def main(self): 
        """ Loads screen and handles traversing using predestined path """
        clock = pygame.time.Clock()
        self.screen.fill((255, 255, 255))

        self.draw_all()
        while 1:
            clock.tick(100)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    # closes program 
                    return
                elif event.type == pygame.KEYDOWN:
                    # resets robot and traverses path 
                    self.robot.reset()
                    self.traversePath()

            self.draw_all()
                    
    def draw_all(self):
        """ Draws all obstacles, start/finish points, and robot """
        for obs in self.obstacles:
            obs.draw()
        pygame.draw.circle(self.screen, "blue", self.start[0:2], 3)
        pygame.draw.circle(self.screen, "blue", self.end[0:2], 3)
        self.robot.draw()
    
    def create_graph(self, shift, angle):
        """ 
        Given a shift and angle to discretize by, will create a 3D "graph"
        Graph is really just a 3D array that will be operated upon with BFS.
        Each "layer" of this array represents a different angle for the robot
        while the 2D arrays of each layer are a representation of the shifts
        to each discretized point. 

        Runtime Analysis:

        Define...
        cols (c) -> length / shift)
        rows (r) -> width / shift)
        depth (d) -> 360 / angle
        n -> size of robot
        m -> size of all obstacles 

        O(d*c*r) to create a model for each point
            -> isValid()
                -> minkowskiSum(): O(n + m)
                -> inside_convex_polygon: O(n + m)
        
        Total: O(d*c*r*(n+m)) which is optimal when discretizing 
        """

        def addBufferZone():
            """
            TODO:
            * modify Minkowski Sum function and implmenetation
            * Implement add buffer zone (this might be tough)
            
            """
            pass
        
        def inside_convex_polygon(p, v):
            """
            p: vertex
            v: list of vertices of polygon
            
            Checks if p is inside of v in O(|v|)-time 
            
            Returns True if inside, False if outside 
            """
            prev_side = None
            for i in range(len(v)):
                a, b = v[i], v[(i+1)%len(v)]
                segment_to_point = (p[0] - a[0], p[1] - a[1])
                segment_of_vertices = (b[0] - a[0], b[1] - a[1])

                # take determinant 
                det = segment_of_vertices[0] * segment_to_point[1] - \
                    segment_of_vertices[1] * segment_to_point[0]

                # check if vertex is on the same side of lines...
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

            # made it out of loop, must be inside 
            return True
        
        def isValid(robot, obstacles):
            """
            Will determine if the robot is valid with respect to the obstacles
            (in other words, check if robot is NOT in an obstacle -- an application
            of the Minkowski Sum!)

            We will get the MinkowskiSum of P and -Q for each Q in Q_list
            and then check whether the resulting polygon contains (0, 0) using 
            inside_convex_polygon(). This effectively checks if P and Q are intersecting
            by which we can determine if the robot positioning is valid.
            """

            # Check boundaries first...
            for pt in robot.getPoints():
                if pt[0] < 0 or pt[1] < 0 or pt[0] > self.w or pt[1] > self.l: 
                    return False
            
            # use minkowski sum to check for intersecting polygons quickly!
            for obsObj in obstacles:
                obs = obsObj.getPoints().copy()
                for i in range(len(obs)):
                    obs[i] = [-obs[i][0], -obs[i][1]]
                ms = minkowskiSum(robot.getPoints().copy(), obs)
                if(inside_convex_polygon([0,0], ms)):
                    return False
            
            # if made it out of loop, robot is in a valid location! return True
            return True

            """
            For a point robot, just use inside_convex_polygon() without
            Minokwski Sum first...

            checks if a point is in the obstacles
            p = robot.getCenter()
            for obs in obstacles:
                if(inside_convex_polygon(p, obs.getPoints())):
                    return False
            return True

            """

        def minkowskiSum(P, Q):
            """
            P: Robot polygon vertices
            Q: Obstacle polygon vertices

            Calculates the vertices of the Minkowski Sum (ms) using an iterative
            method that "cycles" around both shapes using pointers incrementing the 
            pointer with the smaller cumulative polar angle (as used in class). Each cycle
            we add the sum of the vectors at each pointer.

            Method takes O(|P| + |Q|) as each value is visited only once. 
            """
            def reorder_pts(poly):
                """Rotates vertices in polygon array with the bottom-left-most
                value as the first entry. Maintains CCW nature. """
                i_smallest = 0
                for i in range(1, len(poly)):
                    if(poly[i][1] < poly[i_smallest][1] or \
                        (poly[i][1] == poly[i_smallest][1] and poly[i][0] < poly[i_smallest][0])):
                        i_smallest = i
                return poly[i_smallest:] + poly[:i_smallest]

            # preprocessing 
            P = reorder_pts(P)
            Q = reorder_pts(Q)
            P.append(P[0])
            P.append(P[1])
            Q.append(Q[0])
            Q.append(Q[1])

            # initialize loop variables 
            i = j = 0
            ms = []
            while(i < (len(P) - 2) or j < (len(Q) - 2)):
                # add to minkowski sum 
                ms.append([P[i][0] + Q[j][0], P[i][1] + Q[j][1]])

                # calculate segments to compare polar angles 
                P_seg = [P[i+1][0] - P[i][0], P[i+1][1] - P[i][1]]
                Q_seg = [Q[j+1][0] - Q[j][0], Q[j+1][1] - Q[j][1]]
                # cross product 
                cross = P_seg[0] * Q_seg[1] - P_seg[1] * Q_seg[0]
                if(cross >= 0.0):
                    j+=1
                if(cross <= 0.0):
                    i+=1

            return ms
        
        # pre processing: make a new robot to move around 
        robot_pts = self.robot.getPoints().copy()
        robot_center = self.robot.getCenter().copy()
        robot = Robot(robot_center, robot_pts, self.screen)
        # "zero" robot to look at every point
        robot.move(-(robot.getCenter()[0]), -(robot.getCenter()[1]))

        # initailze loop variables 
        a = 0
        cols = math.ceil(self.l / shift)
        rows = math.ceil(self.w / shift)
        depth = math.ceil(360 / angle)
        adjacencyMatrix = []

        for k in range(depth):
            # for each angle...
            adjacencyMatrix.append([])
            robot.rotate(angle)
            for c in range(cols):
                # for each column...
                adjacencyMatrix[k].append([])
                for r in range(rows):
                    # for each robot ...
                    if isValid(robot, self.obstacles):
                        # if in valid location, add to graph marking it as unread (False)
                        adjacencyMatrix[k][c].append([r * shift, c * shift, a, False])
                    else:
                        # if in valid location, add to graph marking it as read (True)
                        adjacencyMatrix[k][c].append([r * shift, c * shift, a, True])
                    robot.move(shift, 0)
                robot.move(-(shift * rows), shift)
            robot.move(0, -(shift * cols))
            a+=angle

        # Save adjacency matrix and find path
        self.adjacencyMatrix = adjacencyMatrix
        self.path = self.bfsPathFind(shift, angle)

    def bfsPathFind(self, shift, angle):
        """
        Finds a path using the graph (adjacencyMatrix) using a BFS approach.
        Will find the shortest path (prioritizing rotation first). If no path
        exists, will return -1.
        
        Runtime for path-find queries: O(r*c*d)
        """
        
        # create copy (as this array will be edited)
        adjacencyMatrix = self.adjacencyMatrix.copy()
        self.shift = shift
        self.angle = angle

        # BFS preprocessing
        queue = []
        s = self.start.copy()
        queue.append([[s[2]//angle, s[1]//shift, s[0]//shift]])
        adjacencyMatrix[s[2]//angle][s[1]//shift][s[0]//shift][3] = True
        dirs = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]

        # start loop using a queue
        while(queue):
            # pop from queue
            path = queue.pop(0)

            # take the most recent location in path to check 
            x_i, y_i, z_i = path[-1]

            # get the assocaited orientation at that location 
            node_val = adjacencyMatrix[x_i][y_i][z_i]
            x, y, z, v = node_val

            # if we found the end, return the path 
            if(node_val[0:3] == self.end):
                return path
            
            # else search in 6 more directions (prioritizing angle)
            for a, b, c in dirs:
                # make sure next point is valid and UNREAD (valid)
                if(z + a * angle >= 0 and z + a * angle < 360
                and y + b * shift >= 0 and y + b * shift < self.l
                and x + c * shift >= 0 and x + c * shift < self.w and not
                adjacencyMatrix[x_i+a][y_i+b][z_i+c][3]):
                    # mark as read
                    adjacencyMatrix[x_i+a][y_i+b][z_i+c][3] = True
                    # add to path and append the new path to check
                    newpath = path.copy()
                    newpath.append([x_i+a,y_i+b,z_i+c])
                    queue.append(newpath)
        # if path could not be found 
        return -1

    def traversePath(self):
        """
        Traverses a path using self.path. Self.path holds the values, but
        the robot along this path takes some work. 
        """
        if(self.path == -1):
            print("No path found")
            return
        
        # set start point to previous values 
        x_p, y_p, a_p = self.start
        for x_i, y_i, z_i in self.path:
            time.sleep(self.shift*0.01)
            x, y, a, v = self.adjacencyMatrix[x_i][y_i][z_i]

            # move robot along path and redraw all points
            self.robot.move(x - x_p, y - y_p)
            self.robot.rotate(a - a_p)
            self.screen.fill((255, 255, 255))
            self.draw_all()
            # set previous to current values
            x_p = x
            y_p = y
            a_p = a


def main():
    # initiate simulator and run it 
    pygame.init()
    pygame.display.set_caption("Robot Simulation Part 1")
    screen = pygame.display.set_mode((640, 480)) #width, length
    simulation = Simulator(screen, 640, 480) #width, length
    print("Loading Path...")
    simulation.create_graph(5, 2) #shift, angle
    simulation.main()
main()