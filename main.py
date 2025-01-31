from classes import Bucket
from classes import Obstacle
from classes import Goal
from collect_coords import capture_coords
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
from matplotlib.animation import FuncAnimation
from collections import defaultdict
import pickle

matplotlib.use('TkAgg')

class Simulation:
    def __init__(self, pointCloud=[], initial_tip=(3,3)):
        # Create some sample objects
        self.bucket = Bucket(initial_tip=initial_tip)
        self.obstacle = Obstacle(dimensions=(4, 3), CoM=(8, 8))
        self.goal = Goal(r=2, CoM=(15, 15))  
        self.iterations = 0

        if len(pointCloud) == 0:
            self.pointCloud = np.array(capture_coords())
        else:
            self.pointCloud = pointCloud

        self.amount_material = len(self.pointCloud)

        self.reverse_dirs = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W', 'NE': 'SW', 'SW': 'NE', 'SE': 'NW', 'NW': 'SE'}

        self.currentQ = 0.0
        # Initialize Q-table as a dictionary of float key-value pairs
        self.Q = defaultdict(float)

        # Plot the objects
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        # Get the DPI (dots per inch) of the figure
        dpi = self.fig.dpi

        # Get the dimensions of the figure in pixels
        fig_width, fig_height = self.fig.get_size_inches() * dpi

        # The scaling factors
        scale_x = 20 / fig_width  # Scale factor for x coordinates (from 0-20 range)
        scale_y = 20 / fig_height  # Scale factor for y coordinates (from 0-20 range)

        # Convert pixel coordinates to normalized mpl coordinates
        self.pointCloud[:, 0] = self.pointCloud[:, 0] * scale_x
        self.pointCloud[:, 1] = (fig_height - self.pointCloud[:, 1]) * scale_y  # Flip y-axis to match mpl coordinate system

    def reward_fxn(self, inside_points):
        x = self.bucket.tipx
        y = self.bucket.tipy
        reward = 0
        # Make reward of successfully scooping new material proportional
        # to the amount of material currently on the board to incentivize
        # gathering the last few pieces before returning to goal
        if len(inside_points) != 0:
            for _ in inside_points:
                reward += 100 / len(self.pointCloud)
                print('reward for collecting material at each point', (x,y), ':', 1000 / len(self.pointCloud))
        # Hitting an obstacle should be discincentivized no matter how
        # much material has been scooped
        if self.bucket.isCollision == True:
            print('penalty for hitting obstacle', -50)
            reward -= 50
        if self.bucket.isGoal == True:
            # Incentivize arriving to goal
            print('reward for reaching goal', 100)
            reward += 100
        else:
            # Discincentivize being far from the goal
            dist_from_goal = np.sqrt((self.goal.CoM[0] - x)**2 + (self.goal.CoM[1] - y)**2)
            print('dist from goal reward', 1 / dist_from_goal)
            reward +=  1 / dist_from_goal

        return reward
    
    def action_value_fxn(self, alpha, gamma, inside_points, transition_probs):
        reward = self.reward_fxn(inside_points)
        return (1 - alpha) * self.currentQ + alpha * (reward + gamma * transition_probs)
    
    def update_table(self, state, action):
        self.Q[state] += float(np.round(action, 2))  # Assigning Q-value
    
    def get_state(self):
        distances = [np.linalg.norm(point - np.array((self.bucket.tipx, self.bucket.tipy))) for point in self.pointCloud]
        # Find index of minimum distance
        min_index = np.argmin(distances)
        closest_point = self.pointCloud[min_index]  # The corresponding point
        current_state = {'agent location': self.bucket.getVertices(),
                         'obstacle location': (self.obstacle.CoM, (self.obstacle.dimx, self.obstacle.dimy)),
                         'nearest reward': closest_point}
        return current_state

    def init_animation(self):

        # Plot the constructed point cloud
        self.scatter = self.ax.scatter(self.pointCloud[:, 0], self.pointCloud[:, 1], color='black')

        # Plot the bucket
        self.bucket_polygon = Polygon(self.bucket.getVertices(), 
                                closed=True,
                                edgecolor='blue',
                                facecolor='lightblue'
                                )
        self.ax.add_patch(self.bucket_polygon)
        self.bucket_tip_plot = self.ax.scatter(self.bucket.tipx, self.bucket.tipy, color='purple')

        # Plot the obstacle as a rectangle
        self.obstacle_rect = Rectangle(
            (self.obstacle.CoM[0] - self.obstacle.dimx / 2, self.obstacle.CoM[1] - self.obstacle.dimy / 2),
            self.obstacle.dimx,
            self.obstacle.dimy,
            edgecolor='red',
            facecolor='tomato'
        )
        self.ax.add_patch(self.obstacle_rect)

        # Plot the goal as a circle
        self.goal_circle = Circle(
            self.goal.CoM,
            self.goal.radius,
            edgecolor='green',
            facecolor='lightgreen'
        )
        self.ax.add_patch(self.goal_circle)

        # Set axis limits
        plt.xlim(0, 20)
        plt.ylim(0, 20)

        # Add grid, legend, and labels
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper left')
        plt.title('2D Simulation of Bucket, Obstacle, and Goal')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')

        plt.show()

        return self.bucket_polygon, self.bucket_tip_plot, self.obstacle_rect, self.goal_circle

    def update(self, frame, max_iterations=100):
        while self.iterations < max_iterations:
            # Set the current direction for each update
            directions = ['N', 'S', 'W', 'E', 'NW', 'NE', 'SW', 'SE']
            self.current_dir = random.choice(directions)
            # Move bucket 1 step in existing direction
            self.bucket.move(velocity=1.0, time=1.0, current_dir=self.current_dir)
            # Update the polygon vertices
            self.bucket_polygon.set_xy(self.bucket.getVertices())
            # Update the bucket tip marker
            self.bucket_tip_plot.set_offsets((self.bucket.tipx, self.bucket.tipy))
            if self.bucket.tipx <= 0 or self.bucket.tipy <= 0:
                print('out of bounds !')
                # Find previous direction to travel back in
                prev_dir = self.reverse_dirs[self.current_dir]
                # Move bucket 1 step back in previous direction
                self.bucket.move(velocity=1.0, time=1.0, current_dir=prev_dir)
                # Re-update the polygon vertices
                self.bucket_polygon.set_xy(self.bucket.getVertices())
                # Re-update the bucket tip marker
                self.bucket_tip_plot.set_offsets((self.bucket.tipx, self.bucket.tipy))                 
            inside_points = self.bucket.gatherMaterial(self.pointCloud)
            if len(inside_points) != 0:
                # Remove all elements in the sublist
                self.pointCloud = np.array([point for point in self.pointCloud.tolist() if not np.any(np.all(point == inside_points, axis=1))])
                self.scatter.set_offsets(self.pointCloud)
            self.iterations += 1
            self.bucket.reachedGoal(self.goal)
            if self.bucket.isGoal == True:
                print('Goal succesfully reached!')
                self.ani.event_source.stop()  # Stop the animation
                reward = self.reward_fxn(inside_points)
                # Update the Q-table
                (x, y) = (self.bucket.tipx, self.bucket.tipy)
                self.update_table((x, y), reward)
                break
            else:
                self.bucket.checkCollison(self.obstacle)
                reward = self.reward_fxn(inside_points)
                if self.bucket.isCollision == True:
                    print('collided !')
                    # Find previous direction to travel back in
                    prev_dir = self.reverse_dirs[self.current_dir]
                    # Move bucket 1 step back in previous direction
                    self.bucket.move(velocity=1.0, time=1.0, current_dir=prev_dir)
                    # Re-update the polygon vertices
                    self.bucket_polygon.set_xy(self.bucket.getVertices())
                    # Re-update the bucket tip marker
                    self.bucket_tip_plot.set_offsets((self.bucket.tipx, self.bucket.tipy)) 
                    # Set collision flag back to False
                    self.bucket.isCollision = False
                print('iter number', self.iterations)
            # Update the Q-table
            (x, y) = (self.bucket.tipx, self.bucket.tipy)
            self.update_table((x, y), reward)

            return self.bucket_polygon, self.bucket_tip_plot
    
    def animate(self):
        # Animate the bucket
        self.ani = FuncAnimation(self.fig, self.update, frames=200, init_func=self.init_animation, interval=50, blit=False)
        plt.show()

        return self.ani

if __name__ == '__main__':
    simulation = Simulation()
    simulation.animate()
    print('finished animating?')
