import pygame
import math
import numpy as np
import random
from matplotlib.path import Path
from matplotlib.transforms import Bbox

class Bucket:

    def __init__(self, initial_tip):
        self.tipx = initial_tip[0]
        self.tipy = initial_tip[1]

        self.gathered_material = []

        self.isCollision = False
        self.isGoal = False

    def gatherMaterial(self, pointCloud):
        # Define the vertices of the polygon
        vertices = self.getVertices()
        # Create a Polygon object
        polygon_path = Path(vertices)
        # Check which points are inside the polygon
        contained = polygon_path.contains_points(pointCloud)

        # Filter points that are inside
        inside_points = pointCloud[contained]
        if len(inside_points) != 0:
            self.gathered_material.extend(inside_points.tolist())
        return inside_points

    def checkCollison(self, obstacle):
        vertices = np.array(self.getVertices())
        vertices_obs = obstacle.getVertices()
        x_min, y_min = vertices.min(axis=0)
        x_max, y_max = vertices.max(axis=0)
        x_min_obs, y_min_obs = vertices_obs.min(axis=0)
        x_max_obs, y_max_obs = vertices_obs.max(axis=0)
        bbox = Bbox.from_extents(x_min, y_min, x_max, y_max)
        bbox_obs = Bbox.from_extents(x_min_obs, y_min_obs, x_max_obs, y_max_obs)
        if ((x_min <= x_min_obs and x_min_obs <= x_max) and (y_min <= y_max_obs and y_max_obs <= y_max)) or \
        (x_min_obs <= x_min and x_min <= x_max_obs) and (y_min_obs <= y_max and y_max <= y_max_obs):
            self.isCollision = True

    def reachedGoal(self, goal):
        distance = np.sqrt((self.tipx - goal.CoM[0]) ** 2 + (self.tipy - goal.CoM[1]) ** 2)
        if distance < goal.radius:
            self.isGoal = True
        
    def getVertices(self):
        return [(self.tipx, self.tipy), (self.tipx - 3, self.tipy + 1), (self.tipx - 3, self.tipy + 3), (self.tipx, self.tipy + 3)]
    
    def move(self, velocity, time, current_dir):
        if current_dir == 'N':
            self.tipy += velocity * time
        elif current_dir == 'S':
            self.tipy -= velocity * time
        elif current_dir == 'E':
            self.tipx += velocity * time
        elif current_dir == 'W':
            self.tipx -= velocity * time
        elif current_dir == 'NW':
            self.tipy += velocity * time
            self.tipx -= velocity * time
        elif current_dir == 'NE':
            self.tipy += velocity * time
            self.tipx += velocity * time
        elif current_dir == 'SW':
            self.tipy -= velocity * time
            self.tipx -= velocity * time
        elif current_dir == 'SE':
            self.tipy -= velocity * time
            self.tipx += velocity * time


class Obstacle:

    def __init__(self, dimensions, CoM):
        self.dimx = dimensions[0]
        self.dimy = dimensions[1]
        self.CoM = CoM

    def getVertices(self):
        vertices = np.array([(self.CoM[0] - self.dimx / 2, self.CoM[1] - self.dimy / 2),
                             (self.CoM[0] + self.dimx / 2, self.CoM[1] + self.dimy / 2),
                             (self.CoM[0] - self.dimx / 2, self.CoM[1] + self.dimy / 2),
                             (self.CoM[0] + self.dimx / 2, self.CoM[1] - self.dimy / 2)])
        return vertices


class Goal:

    def __init__(self, r, CoM):
        self.radius = r
        self.CoM = CoM

    
        
