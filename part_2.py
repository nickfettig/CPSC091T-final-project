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
from shapely.geometry import LineString

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

    def drawBuffer(self):
        pygame.draw.polygon(self.surface, "orange", self.points)
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
            [[30, 70], [20, 100], [30, 130], [50, 130], [60, 100], [50, 70]], screen)
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

        """ Pushing robots """
        self.pr_amount = None
        self.pr_size = None
        self.buffer = None

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
    
    def initialize_pushers(self, amount, size):
        self.pr_amount = amount
        self.pr_size = size

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

        def addBufferZone(pts):
            """
            TODO:
            * modify Minkowski Sum function and implmenetation
            * Implement add buffer zone (this might be tough)
            
            """

            # check that pushing robot parameters are specified 
            if(self.pr_amount == None or self.pr_size == None):
                return

            pr_square = [[-self.pr_size*3, -self.pr_size*3],
                [-self.pr_size*3, self.pr_size*3],
                [self.pr_size*3, self.pr_size*3],
                [self.pr_size*3, -self.pr_size*3]]

            ms = self.minkowskiSum(pts, pr_square)
            self.buffer = ms.copy()

            return ms
        
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
                ms = self.minkowskiSum(robot.getPoints().copy(), obs)
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
        
        # pre processing: make a new robot to move around 
        robot_pts = self.robot.getPoints().copy()
        robot_center = self.robot.getCenter().copy()

        buff = addBufferZone(robot_pts)
        robot = Robot(robot_center, buff, self.screen)

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
        bufferRobot = Robot(self.robot.getCenter().copy(),
        self.buffer.copy(), self.screen)
        curr_dir = None

        path = self.path.copy()
        path.pop(0) # clear first value
        task = None
        
        while(path):
            if(task != "reposition"):
                x_i, y_i, z_i = path.pop(0)

            print(x_i, y_i, z_i)
            print(curr_dir)
            time.sleep(self.shift*0.01)
            x, y, a, v = self.adjacencyMatrix[x_i][y_i][z_i]

            # DETERMINE TASK
            if(a != a_p):
                task = "rotate"
                print("rotate")
                self.robot.rotate(a - a_p)
                bufferRobot.rotate(a - a_p)
                a_p = a
            elif(x_p != x and curr_dir == "x"):
                task = "push"
                print("push")
                self.robot.move(x - x_p, y - y_p)
                bufferRobot.move(x - x_p, y - y_p)
                x_p = x
            elif(y_p != y and curr_dir == "y"):
                task = "push"
                print("push")
                self.robot.move(x - x_p, y - y_p)
                bufferRobot.move(x - x_p, y - y_p)
                y_p = y
            else:
                task = "reposition"
                print("reposition")
                if(x_p != x):
                    curr_dir = "x"
                else:
                    curr_dir = "y"

            # move robot along path and redraw all points
            self.screen.fill((255, 255, 255))

            # obstacles and polygons
            bufferRobot.drawBuffer()

            # construct repositioning path TODO: testing
            self.constructRepositioningPath(1, 2, self.robot.getPoints().copy())

            #TODO: TESTING STABLIZER
            stablizer = Stablizer(self.robot.getCenter().copy(), self.robot.getPoints().copy(), \
            self.buffer.copy(), [0,0], self.pr_size, self.screen)
            
            self.draw_all()

    def minkowskiSum(self, P, Q):
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

    def constructRepositioningPath(self, p1, p2, M):

        path_endpoint_box = [[-self.pr_size*1.5, -self.pr_size*1.5],
            [-self.pr_size*1.5, self.pr_size*1.5],
            [self.pr_size*1.5, self.pr_size*1.5],
            [self.pr_size*1.5, -self.pr_size*1.5]]

        path_endpoints = self.minkowskiSum(M, path_endpoint_box)

        n = len(path_endpoints)
        for i in range(n):
            pygame.draw.line(self.screen, "blue", \
                path_endpoints[i], path_endpoints[(i+1)%n], 1)


class Stablizer():
        def __init__(self, C, M, M_p, N, pr_size, screen):
            # won't change
            self.surface = screen
            self.pr_size = pr_size

            # intially stabilize to instatiate variables
            self.restablize(C, M, M_p, N)

        def restablize(self, C, M, M_p, N):
            def line(p1, p2):
                A = (p1[1] - p2[1])
                B = (p2[0] - p1[0])
                C = (p1[0]*p2[1] - p2[0]*p1[1])
                return A, B, -C

            """ https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect """
            def intersection(LS1, LS2):
                if(not LineString(LS1).intersects(LineString(LS2))):
                    return -1
                L1 = line(LS1[0], LS1[1])
                L2 = line(LS2[0], LS2[1])
                D  = L1[0] * L2[1] - L1[1] * L2[0]
                Dx = L1[2] * L2[1] - L1[1] * L2[2]
                Dy = L1[0] * L2[2] - L1[2] * L2[0]
                if D != 0:
                    x = Dx / D
                    y = Dy / D
                    return x,y
                else:
                    return -1

            def find_edge_target_point(edge, C):
                dir_perp_edge = [-(edge[1][1] - edge[0][1]), (edge[1][0] - edge[0][0])]
                perp_edge_norm = np.linalg.norm(dir_perp_edge)
                perp_edge = [C, [C[0] + dir_perp_edge[0] * 2 * self.S_r / perp_edge_norm, \
                    C[1] + dir_perp_edge[1] * 2 * self.S_r / perp_edge_norm]]
                intersection_pt = intersection(edge, perp_edge)

                # find associated target rest point
                rest_pt = [intersection_pt[0] + dir_perp_edge[0] * 2.5 * self.pr_size / perp_edge_norm, \
                    intersection_pt[1] + dir_perp_edge[1] * 2.5 * self.pr_size / perp_edge_norm]

                return edge, intersection_pt, rest_pt

            S_r = 0
            for pt in M:
                self.S_r = S_r = max(S_r, math.sqrt((C[0]-pt[0])**2 + (C[1]-pt[1])**2))
            dir_L2 = [C[0] - N[0], C[1] - N[1]]
            # extend L2 by 2 * S_r to ensure rays go through boundary of M
            L2_norm = np.linalg.norm(dir_L2)
            L2 = [dir_L2[0] * 2 * S_r / L2_norm, dir_L2[1] * 2 * S_r / L2_norm]

            dir_L3 = [-dir_L2[1], dir_L2[0]]
            L3_norm = np.linalg.norm(dir_L3)
            L3 = [dir_L3[0] * 2 * S_r / L3_norm, dir_L3[1] * 2 * S_r / L3_norm]

            # convert to rays centered at C
            L1 = self.L1 = [C, [C[0] - L2[0], C[1] - L2[1]]]
            L2 = self.L2 = [C, [C[0] + L2[0], C[1] + L2[1]]]
            L4 = self.L4 = [C, [C[0] - L3[0], C[1] - L3[1]]]
            L3 = self.L3 = [C, [C[0] + L3[0], C[1] + L3[1]]]

            # pygame.draw.line(self.surface, "green", \
            #     L2[0], L2[1], 1)

            for i in range(len(M)):
                edge = [M[i], M[(i+1)%len(M)]]
                if(intersection(L2, edge) != -1):
                    e2, pu, pr = find_edge_target_point(edge, C)
                    self.e2 = e2
                    self.pu = pu
                    self.pr = pr
                if(intersection(L3, edge) != -1):
                    e3, la, lar = find_edge_target_point(edge, C)
                    self.e3 = e3
                    self.la = la
                    self.lar = lar

                    # TODO: FIX THESE: POTENTIALLY NOT CORRECT
                    r0 = self.r0 = e3[0]
                    l0 = self.l0 = e3[1]
                if(intersection(L4, edge) != -1):
                    e4, ra, rar = find_edge_target_point(edge, C)
                    self.e4 = e4
                    self.ra = ra
                    self.rar = rar
                
            #TODO: ADD ROTATION TARGET REST / BALANCE POINTS HERE

def main():
    # initiate simulator and run it 
    pygame.init()
    pygame.display.set_caption("Robot Simulation Part 2")
    screen = pygame.display.set_mode((640, 480)) #width, length
    simulation = Simulator(screen, 640, 480) #width, length
    simulation.initialize_pushers(3, 6) #amount, size (diameter)
    print("Loading Path...")
    simulation.create_graph(10, 5) #shift, angle
    print("Path loaded.")
    simulation.main()
main()