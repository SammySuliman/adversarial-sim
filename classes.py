import pygame
import math
import numpy as np
import random
from matplotlib.path import Path

class Bucket:

    def __init__(self, initial_tip):
        self.tipx = initial_tip[0]
        self.tipy = initial_tip[1]

        self.gathered_material = []

    def gatherMaterial(self, pointCloud):
        # x_coords = pointCloud[:, 0]
        # y_coords = pointCloud[:, 1]
        # Define the vertices of the polygon
        vertices = self.getVertices()
        # Create a Polygon object
        polygon_path = Path(vertices)
        # Check which points are inside the polygon
        contained = polygon_path.contains_points(pointCloud)

        # Filter points that are inside
        inside_points = pointCloud[contained]
        # print('inside pts', inside_points)
        if len(inside_points) != 0:
            self.gathered_material.extend(inside_points.tolist())
        return inside_points
        '''
        if self.tipx in x_coords:
            indices = np.argwhere(pointCloud[:, 0] == self.tipx)
            for index in indices:
                if self.tipy in pointCloud[index][:, 1]:
                    self.gathered_material.append((self.tipx, self.tipy))
        '''

    def checkCollison(self, obstacle):
        if self.tipx > (obstacle.CoM[0] - obstacle.dimx / 2) and self.tipx < (obstacle.CoM[0] + obstacle.dimx / 2):
            if self.tipy > (obstacle.CoM[1] - obstacle.dimy / 2) and self.tipy < (obstacle.CoM[1] + obstacle.dimy / 2):
                return True
            return False
        return False
    
    def reachedGoal(self, goal):
        distance = np.sqrt((self.tipx - goal.CoM[0]) ** 2 + (self.tipy - goal.CoM[1]) ** 2)
        if distance < goal.radius:
            return True
        else:
            return False
        
    def getVertices(self):
        return [(self.tipx, self.tipy), (self.tipx - 3, self.tipy + 1), (self.tipx - 3, self.tipy + 3), (self.tipx, self.tipy + 3)]
    
    def move(self, velocity, time, current_dir):
        if current_dir == 'N':
            self.tipy += velocity * time
            #print('tip y', self.tipy)
        elif current_dir == 'S':
            self.tipy -= velocity * time
            #print('tip y', self.tipy)
        elif current_dir == 'E':
            self.tipx += velocity * time
            #print('tip x', self.tipx)
        elif current_dir == 'W':
            self.tipx -= velocity * time
            #print('tip x', self.tipx)
        elif current_dir == 'NW':
            self.tipy += velocity * time
            self.tipx -= velocity * time
            #print('tip x', self.tipx)
            #print('tip y', self.tipy)
        elif current_dir == 'NE':
            self.tipy += velocity * time
            self.tipx += velocity * time
            #print('tip x', self.tipx)
            #print('tip y', self.tipy)
        elif current_dir == 'SW':
            self.tipy -= velocity * time
            self.tipx -= velocity * time
            #print('tip x', self.tipx)
            #print('tip y', self.tipy)
        elif current_dir == 'SE':
            self.tipy -= velocity * time
            self.tipx += velocity * time
            #print('tip x', self.tipx)
            #print('tip y', self.tipy)


class Obstacle:

    def __init__(self, dimensions, CoM):
        self.dimx = dimensions[0]
        self.dimy = dimensions[1]
        self.CoM = CoM

class Goal:

    def __init__(self, r, CoM):
        self.radius = r
        self.CoM = CoM

    
        
