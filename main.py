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

class Robot():
    def __init__(self, x, y, points, screen):
        self.center = [x, y]
        self.points = points
        self.surface = screen
    
    def set(self, x, y,orientation): #setter
        self.x = x
        self.y = y
        self.orientation = orientation

    def move(self, x, y):
        curr_x, curr_y = self.center 
        self.center = [curr_x + x, curr_y + y]
        for i in range(len(self.points)):
            curr_x, curr_y = self.points[i]
            self.points[i] = [curr_x + x, curr_y + y]

    def rotate(self, angle):
        pass
        #TODO after translation works

    def draw(self):
        pygame.draw.polygon(self.surface, "black", self.points)
        pygame.draw.circle(self.surface, "red", self.center, 3)
        pygame.display.flip()

class Simulator():
    def __init__(self, screen):
        self.screen = screen
        self.robot = Robot(20, 100, \
            [[10, 50], [10, 150], [60, 150], [60, 50]], screen)

    def main(self):
        clock = pygame.time.Clock()
        while 1:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                elif event.type == pygame.KEYDOWN:
                    self.robot.move(10, 10)
            self.screen.fill((255, 255, 255))
            self.robot.draw()

def main():
    pygame.init()
    pygame.display.set_caption("Robot Simulation")
    screen = pygame.display.set_mode((640,480))
    simulation = Simulator(screen)
    simulation.main()
main()