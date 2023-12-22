"""
Nicholas Fettig
CPSC 091T: Computational Geometry
Prof. Neil Lutz
Fall 2023

Final Project:
Rigorous Movement of Convex Polygons on a Path Using Multiple Robots Simulation

DESCRIPTION

pip libs:
pygame, numpy, shapely

"""

import pygame
import numpy as np
import math
import time
from shapely.geometry import LineString


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

class Obstacle():
    """ Defines the obstacle positions """

    def __init__(self, points, screen):
        self.points = points
        self.surface = screen 
    
    def draw(self):
        pygame.draw.polygon(self.surface, "red", self.points)

    def getPoints(self):
        return self.points

class Object():
    """ 
        Handles robot positioning, movement, and rotation
        Note that all operations are local and in-place
    """
    def __init__(self, center, points, screen):
        self.center = center
        self.points = points.copy()
        self.surface = screen
        self.initial_center = center
        self.initial_points = points.copy()

    def move(self, x, y):
        """ Will move the robot a set x and y from its current location """
        curr_x, curr_y = self.center 
        self.center = [curr_x + x, curr_y + y]
        pt_accum = []
        for i in range(len(self.points)):
            curr_x, curr_y = self.points[i]
            pt_accum.append([curr_x + x, curr_y + y])
        self.points = pt_accum

    def draw(self):
        """ Draws robot to screen """
        pygame.draw.polygon(self.surface, "black", self.points)
        pygame.draw.circle(self.surface, "red", self.center, 3)
    
    def drawBuffer(self):
        """ Draws robot to screen """
        pygame.draw.polygon(self.surface, "orange", self.points)
    
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
        self.points = self.initial_points
        self.center = self.initial_center

class Simulator():
    """ 
    Handles operation of the simulation. Controlls robot / obstacles 
    and holds all necessary algorithms.
    """
    def __init__(self, screen, w, l):
        self.screen = screen
        self.start = [40, 100]
        self.end = [560, 400]
        self.w = w
        self.l = l

        """ 
        Play around with these values to test different robots through courses!

        NOTE: ALL POINTS MUST BE IN CCW ORDER FOR MINKOWSKI SUM TO WORK
        """
        self.object = Object(self.start, \
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

        """ Pushing robots """
        self.pr_amount = None
        self.pr_size = None
        self.buffer = None

    def main(self): 
        """ Loads screen and handles traversing using predestined path """
        clock = pygame.time.Clock()
        self.screen.fill((255, 255, 255))

        self.bufferObject = Object(self.object.getCenter().copy(),
        self.buffer.copy(), self.screen)
        self.bufferObject.drawBuffer()

        self.robots = []
        for i in range(self.pr_amount):
            self.robots.append(Robot(self.pr_size, [40 * (i+1), self.l-50], self.screen))
            self.robots[i].draw()

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
                    self.object.reset()
                    self.bufferObject.reset()
                    self.traversePath()

            self.draw_all()
                    
    def draw_all(self):
        """ Draws all obstacles, start/finish points, and robot """
        for obs in self.obstacles:
            obs.draw()
        pygame.draw.circle(self.screen, "blue", self.start, 3)
        pygame.draw.circle(self.screen, "blue", self.end, 3)
        self.object.draw()
        pygame.display.update()
    
    def initialize_pushers(self, amount, size):
        self.pr_amount = amount
        self.pr_size = size
    
    def create_graph(self, shift):
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
        robot_pts = self.object.getPoints().copy()
        robot_center = self.object.getCenter().copy()

        buff = addBufferZone(robot_pts)
        robot = Object(robot_center, buff, self.screen)
        # "zero" robot to look at every point
        robot.move(-(robot.getCenter()[0]), -(robot.getCenter()[1]))

        # initailze loop variables 
        cols = math.ceil(self.l / shift)
        rows = math.ceil(self.w / shift)
        adjacencyMatrix = []

        for c in range(cols):
            # for each column...
            adjacencyMatrix.append([])
            for r in range(rows):
                # for each robot ...
                if isValid(robot, self.obstacles):
                    # if in valid location, add to graph marking it as unread (False)
                    adjacencyMatrix[c].append([r * shift, c * shift, False])
                else:
                    # if in valid location, add to graph marking it as read (True)
                    adjacencyMatrix[c].append([r * shift, c * shift, True])
                robot.move(shift, 0)
            robot.move(-(shift * rows), shift)
        robot.move(0, -(shift * cols))

        # Save adjacency matrix and find path
        self.adjacencyMatrix = adjacencyMatrix
        self.path = self.bfsPathFind(shift)

    def bfsPathFind(self, shift):
        """
        Finds a path using the graph (adjacencyMatrix) using a BFS approach.
        Will find the shortest path (prioritizing rotation first). If no path
        exists, will return -1.
        
        Runtime for path-find queries: O(r*c*d)
        """
        
        # create copy (as this array will be edited)
        adjacencyMatrix = self.adjacencyMatrix.copy()
        self.shift = shift

        # BFS preprocessing
        queue = []
        s = self.start.copy()
        queue.append([[s[1]//shift, s[0]//shift]])
        adjacencyMatrix[s[1]//shift][s[0]//shift][2] = True
        dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        # start loop using a queue
        while(queue):
            # pop from queue
            path = queue.pop(0)

            # take the most recent location in path to check 
            x_i, y_i = path[-1]

            # get the assocaited orientation at that location 
            node_val = adjacencyMatrix[x_i][y_i]
            x, y, v = node_val

            # if we found the end, return the path 
            if(node_val[0:2] == self.end):
                return path
            
            # else search in 6 more directions (prioritizing angle)
            for a, b in dirs:
                # make sure next point is valid and UNREAD (valid)
                if(y + b * shift >= 0 and y + b * shift < self.l
                and x + a * shift >= 0 and x + a * shift < self.w and not
                adjacencyMatrix[x_i+a][y_i+b][2]):
                    # mark as read
                    adjacencyMatrix[x_i+a][y_i+b][2] = True
                    # add to path and append the new path to check
                    newpath = path.copy()
                    newpath.append([x_i+a,y_i+b])
                    queue.append(newpath)
        # if path could not be found 
        return -1

    def traversePath(self):
        """
        Traverses a path using self.path. Self.path holds the values, but
        the robot along this path takes some work. 
        """

        def updateMapping():
            self.screen.fill((255, 255, 255))
            self.bufferObject.drawBuffer()        
            for i in self.robots:
                i.draw()
            self.draw_all()

        def constructRepositioningPath(p1, p2, M):

            path_endpoint_box = [[-self.pr_size*1.5, -self.pr_size*1.5],
                [-self.pr_size*1.5, self.pr_size*1.5],
                [self.pr_size*1.5, self.pr_size*1.5],
                [self.pr_size*1.5, -self.pr_size*1.5]]

            path_endpoints = self.minkowskiSum(M, path_endpoint_box)

            n = len(path_endpoints)
            idx_p1 = 0
            idx_p2 = 0
            for i in range(n):
                if(np.linalg.norm([p1[0] - path_endpoints[i][0],
                p1[1] - path_endpoints[i][1]]) < \
                np.linalg.norm([p1[0] - path_endpoints[idx_p1][0],
                p1[1] - path_endpoints[idx_p1][1]])):
                    idx_p1 = i
                if(np.linalg.norm([p2[0] - path_endpoints[i][0],
                p2[1] - path_endpoints[i][1]]) < \
                np.linalg.norm([p2[0] - path_endpoints[idx_p2][0],
                p2[1] - path_endpoints[idx_p2][1]])):
                    idx_p2 = i

            if(abs(idx_p1 - idx_p2) == 1 or \
                ((idx_p1 + 1)%n == 0 and idx_p2 == 0) or
                ((idx_p2 + 1)%n == 0 and idx_p1 == 0)):
                path_lst = []
            elif(idx_p1 < idx_p2):
                path_lst = path_endpoints[idx_p1:(idx_p2+1)]
            elif(idx_p1 > idx_p2):
                path_lst = (path_endpoints[idx_p2:(idx_p1+1)])
                path_lst.reverse()
            else:
                path_lst = []

            path = [p1] + path_lst + [p2]
            return path

        def moveRobot(usingRobot, repo_path, n):
            for repo_pt in repo_path:
                vec = [(repo_pt[0] - usingRobot.position[0])/n, (repo_pt[1] - usingRobot.position[1])/n]
                for i in range(n):
                    time.sleep(0.01)
                    usingRobot.move(vec[0], vec[1], label)
                    updateMapping()

        def useVisibilityGraph(p1, p2):
            obs_box = [[-self.pr_size*0.5, -self.pr_size*0.5],
                [-self.pr_size*0.5, self.pr_size*0.5],
                [self.pr_size*0.5, self.pr_size*0.5],
                [self.pr_size*0.5, -self.pr_size*0.5]]
            
            nodes = {
                tuple(p1): [],
                tuple(p2): []
            }

            for obs in self.obstacles:
                obs_endpoints = self.minkowskiSum(obs.getPoints().copy(), obs_box)
                for ep in obs_endpoints:
                    if ep[0] > 0 and ep[0] < self.w and \
                    ep[1] > 0 and ep[1] < self.l:
                        nodes[tuple(ep)] = []

            for n1 in list(nodes.keys()):
                for n2 in list(nodes.keys()):
                    if(n1 == n2):
                        continue

                    inter = False
                    for o in self.obstacles:
                        obs = o.getPoints()
                        for i in range(len(obs)):
                            edge = [obs[i], obs[(i+1)%len(obs)]]
                            if(intersection([n1, n2], edge) != -1):
                                inter = True
                    if not inter:
                        if(n2 not in nodes[n1]):
                            nodes[n1].append(n2)
                        if(n1 not in nodes[n2]):
                            nodes[n2].append(n1)

            seen = [p1]
            queue = [[p1]]
            while(queue):
                path = queue.pop(0)
                if path[-1] == tuple(p2):
                    return [p1] + path

                for i in nodes[tuple(path[-1])]:
                    if (i not in seen):
                        seen.append(i)
                        newpath = path.copy()
                        newpath.append(i)
                        queue.append(newpath)

            return p1


        if(self.path == -1):
            print("No path found")
            return

        stablizer = Stablizer(self.pr_size, self.screen)

        dangerousRobot = None

        path = self.path.copy()
        path.pop(0)

        for x_i, y_i in path:
            # time.sleep(self.shift*0.01)
            x, y, v = self.adjacencyMatrix[x_i][y_i]

            stablizer_path = stablizer.restablize(self.object.getCenter().copy(),
            self.object.getPoints().copy(), [x,y])
            # move robot along path and redraw all points

            takenRobot = ""
            displacement = [0, 0]
            for pt, label, contactPoint, restPoint in stablizer_path:

                contact = [contactPoint[0] + displacement[0], contactPoint[1] + displacement[1]]
                rest = [restPoint[0] + displacement[0], restPoint[1] + displacement[1]]

                usingRobot = None
                for robot in self.robots:
                    if robot.label == label:
                        usingRobot = robot
                        break
                    if (self.pr_amount == 1 or robot.label != takenRobot) and \
                        (usingRobot == None or np.linalg.norm([robot.position[0] - contact[0], \
                        robot.position[1] - contact[1]]) < \
                        np.linalg.norm([usingRobot.position[0] - contact[0], \
                        usingRobot.position[1] - contact[1]])):
                        usingRobot = robot

                if(abs(pt[0]) < 0.001 and abs(pt[1]) < 0.001):
                    dangerousRobot = usingRobot
                    continue

                takenRobot = label

                if(usingRobot.label == None or usingRobot == dangerousRobot):
                    path_to_robot = useVisibilityGraph(usingRobot.position, self.object.getCenter().copy())
                    rest_of_path = constructRepositioningPath(path_to_robot[-2], contact, self.object.getPoints().copy())
                    repo_path = path_to_robot[:-2] + rest_of_path
                    dangerousRobot = None
                else:
                    repo_path = constructRepositioningPath(usingRobot.position, contact, self.object.getPoints().copy())
                
                if(len(repo_path) > 2):
                    n = 50
                else:
                    n = 4

                moveRobot(usingRobot, repo_path, n)

                time.sleep(0.01)

                self.object.move(pt[0], pt[1])
                self.bufferObject.move(pt[0], pt[1])

                displacement[0] += pt[0]
                displacement[1] += pt[1]

                repo_path = constructRepositioningPath(usingRobot.position, rest, self.object.getPoints().copy())
                moveRobot(usingRobot, repo_path, 10)

                

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
            

class Stablizer():
    def __init__(self, pr_size, screen):
        self.surface = screen
        self.pr_size = pr_size

    def restablize(self, C, M, N):
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

        L2 = self.L2 = [C, [C[0] + L2[0], C[1] + L2[1]]]

        for i in range(len(M)):
            edge = [M[i], M[(i+1)%len(M)]]
            if(intersection(L2, edge) != -1):
                e2, pu, pr = find_edge_target_point(edge, C)
                self.e2 = e2
                self.pu = pu
                self.pr = pr
                e1, la, lar = find_edge_target_point([M[(i-1)%len(M)], M[(i)%len(M)]], C)
                self.e1 = e1
                self.la = la
                self.lar = lar
                e3, ra, rar = find_edge_target_point([M[(i+1)%len(M)], M[(i+2)%len(M)]], C)
                self.e3 = e3
                self.ra = ra
                self.rar = rar


        side = np.sign((N[0] - C[0]) * (-pu[1] - C[1]) - (N[1] - C[1]) * (-pu[0] - C[0]))
        # -1: left, 0: on, 1: right

        v1 = [C[0]-pu[0], C[1]-pu[1]]
        v1_norm = np.linalg.norm(v1)
        v1_unit = [v1[0] / v1_norm, v1[1] / v1_norm]

        if(side == -1):
            v2 = [C[0]-la[0], C[1]-la[1]]
            assist = "la"
            contactPt = la
            restPt = lar
        elif(side == 1):
            v2 = [C[0]-ra[0], C[1]-ra[1]]
            assist = "ra"
            contactPt = ra
            restPt = rar
        else:
            v2 = [0, 0]
            assist = None
            contactPt = None

        v2_norm = np.linalg.norm(v2)
        v2_unit = [v2[0] / v2_norm, v2[1] / v2_norm]

        v3 = [N[0] - C[0], N[1] - C[1]]
        robot_steps = np.linalg.solve([[v1_unit[0], v2_unit[0]], [v1_unit[1], v2_unit[1]]], v3)

        if(robot_steps[0] < 0.001):
            robot_steps[0] = 0
        if(robot_steps[1] < 0.001):
            robot_steps[1] = 0

        step_list = []
        n = 2 # TODO: CAN BE CHANGED (IMPLEMENT LATER)
        v1_step = [robot_steps[0] * v1_unit[0] / n, robot_steps[0] * v1_unit[1] / n]
        v2_step = [robot_steps[1] * v2_unit[0] / n, robot_steps[1] * v2_unit[1] / n]
        for i in range(n):
            step_list.append([v1_step, "pu", pu, pr])
            step_list.append([v2_step, assist, contactPt, restPt])

        return step_list



class Robot():
    def __init__(self, size, p, surface):
        self.size = size
        self.position = p
        self.surface = surface
        self.label = None

    def move(self, dx, dy, label):
        self.position[0] += dx
        self.position[1] += dy
        self.label = label

    def draw(self):
        pygame.draw.circle(self.surface, "blue", self.position, self.size / 2)

                
def main():
    # initiate simulator and run it 
    pygame.init()
    pygame.display.set_caption("Robot Simulation Part 2")
    screen = pygame.display.set_mode((640, 480)) #width, length
    simulation = Simulator(screen, 640, 480) #width, length
    simulation.initialize_pushers(2, 6) #amount, size (diameter)
    print("Loading Path...")
    simulation.create_graph(10) #shift
    print("Path loaded.")
    simulation.main()
main()

"""
Things to do (in priority):
* make faster?
* visbility graph by which the pushers can start in bottom left corner
    -> very important 
* smooth movement of robot / pushers
* work for 3+ robots? 

"""