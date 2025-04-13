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
import pprint

matplotlib.use('TkAgg')

from construct_Q_table import possible_future_states2

class Simulation:
    def __init__(self, pointCloud=[], initial_tip=(3,3), x_bounds=(0,20), y_bounds=(0,20)):
        # Create some sample objects
        self.bucket = Bucket(initial_tip=initial_tip)
        self.obstacle = Obstacle(dimensions=(4, 3), CoM=(8, 8))
        self.goal = Goal(r=2, CoM=(15, 15))  
        self.iterations = 0
        self.iterations_ = 0

        if len(pointCloud) == 0:
            self.pointCloud = np.array(capture_coords())
        else:
            self.pointCloud = pointCloud

        self.amount_material = len(self.pointCloud)

        self.dirs = ['N', 'S', 'W', 'E', 'NE', 'SW', 'NW', 'SE']
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

        print('scale x', scale_x, 'scale y', scale_y)

        # Convert pixel coordinates to normalized mpl coordinates
        self.pointCloud[:, 0] = self.pointCloud[:, 0] * scale_x
        self.pointCloud[:, 1] = (fig_height - self.pointCloud[:, 1]) * scale_y  # Flip y-axis to match mpl coordinate system

        self.x_min, self.x_max  = x_bounds
        self.y_min, self.y_max = y_bounds

        self.explored_directions = defaultdict(list)

        self.reverse_action = None

    def reward_fxn(self, inside_points):
        x = self.bucket.tipx
        y = self.bucket.tipy
        reward = 0

        # Reward scooping new material
        if len(inside_points) != 0:
            for _ in inside_points:
                reward += 10
                print('reward for collecting material at point', (x,y), ':', 10)
        # Disincentivize hitting an obstacle
        if self.bucket.isCollision == True:
            print('penalty for hitting obstacle', -100)
            reward -= 100
        if self.bucket.isGoal == True:
            # Incentivize arriving to goal
            print('reward for reaching goal', 100)
            reward += 100
        else:
            # Incentivize being nearer to the goal
            dist_from_goal = np.sqrt((self.goal.CoM[0] - x)**2 + (self.goal.CoM[1] - y)**2)
            print('dist from goal reward', 1 / dist_from_goal)
            reward +=  1 / dist_from_goal

        return reward
    
    def update_table(self, state, action):
        self.Q[state] += float(np.round(action, 2))  # Assigning Q-value
    
    def choose_best_action(self, Q, state, actions):
        '''We update the animation based on an epsilon-greedy policy, 
           to balance exploration + exploitation'''
        i, j = state
        q_values_for_state = {a: Q[(i, j, a)] for (x, y, a) in Q if (x, y) == (i, j)}
        # print('q values for curr state', q_values_for_state)
        # if multiple actions ave same reward, choose next action randomly
        best_action = max(actions, key=lambda a: Q[(i,j, a)])
        #max_value = Q[(i,j, best_action)]
        #best_actions = [a for a in actions if Q[(i, j, a)] == max_value]
        #best_action = random.choice(best_actions)
        return best_action
    
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
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)

        # Add grid, legend, and labels
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper left')
        plt.title('2D Simulation of Bucket, Obstacle, and Goal')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')

        plt.show()

        return self.bucket_polygon, self.bucket_tip_plot, self.obstacle_rect, self.goal_circle

    def update(self, frame, max_iterations=100):
        directions = self.dirs
        while self.iterations < max_iterations:
            print('directions', directions)
            # Set the current direction for each update
            self.current_dir = random.choice(directions)
            # Move bucket 1 step in existing direction
            self.bucket.move(velocity=1.0, time=1.0, current_dir=self.current_dir)
            # Update the polygon vertices
            self.bucket_polygon.set_xy(self.bucket.getVertices())
            # Update the bucket tip marker
            self.bucket_tip_plot.set_offsets((self.bucket.tipx, self.bucket.tipy))
            if self.bucket.tipx < 0 or self.bucket.tipy < 0 or self.bucket.tipx > 20 or self.bucket.tipy > 20:
                print('out of bounds !')
                # Find previous direction to travel back in
                prev_dir = self.reverse_dirs[self.current_dir]
                # Move bucket 1 step back in previous direction
                self.bucket.move(velocity=1.0, time=1.0, current_dir=prev_dir)
                # Re-update the polygon vertices
                self.bucket_polygon.set_xy(self.bucket.getVertices())
                # Re-update the bucket tip marker
                self.bucket_tip_plot.set_offsets((self.bucket.tipx, self.bucket.tipy))
                # Remove blocked direction from avaliable paths
                directions = [x for x in self.dirs if x != self.current_dir]
            else:
                directions = self.dirs          
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
        
    def update_q(self, frame, max_iterations=200):
        while self.iterations_ < max_iterations:
            print('iter number', self.iterations_)
            # Set the current direction for each update
            pos = (self.bucket.tipx, self.bucket.tipy)
            print('pos', pos)
            directions_dict = possible_future_states2(pos)
            # Ensure bucket cannot travel backwards to previous position
            if pos in directions_dict:
                print('rev action', self.reverse_action)
                if self.reverse_action in directions_dict[pos]:
                    directions_dict[pos].remove(self.reverse_action)  # Remove value from list
                    print('other removed direction', self.reverse_action)  
            # Remove prev explored directions as a choice for this state
            for value in directions_dict[pos]:
                if pos in self.explored_directions and value in self.explored_directions[pos]:  # Ensure value exists in the list
                    directions_dict[pos].remove(value)  # Remove value from list
                    print('removed direction', value)
            # print('directions', directions_dict)
            action = self.choose_best_action(self.Q, pos, directions_dict[pos])
            print('action', action)
            self.reverse_action = self.reverse_dirs[action]
            self.explored_directions[pos].append(action)
            # print('explored directions', self.explored_directions)
            # Move bucket 1 step in existing direction
            self.bucket.move(velocity=1.0, time=1.0, current_dir=action)
            # Update the polygon vertices
            self.bucket_polygon.set_xy(self.bucket.getVertices())
            # Update the bucket tip marker
            self.bucket_tip_plot.set_offsets((self.bucket.tipx, self.bucket.tipy))     
            
            if self.bucket.tipx < 0 or self.bucket.tipy < 0 or self.bucket.tipx > 20 or self.bucket.tipy > 20:
                print('out of bounds !')
                # Find previous direction to travel back in
                prev_dir = self.reverse_dirs[action]
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
            self.bucket.reachedGoal(self.goal)
            if self.bucket.isGoal == True:
                print('Goal succesfully reached!')
                self.ani.event_source.stop()  # Stop the animation
                break
            else:
                self.bucket.checkCollison(self.obstacle)
                if self.bucket.isCollision == True:
                    print('collided !')
                    # Find previous direction to travel back in
                    prev_dir = self.reverse_dirs[action]
                    # Move bucket 1 step back in previous direction
                    self.bucket.move(velocity=1.0, time=1.0, current_dir=prev_dir)
                    # Re-update the polygon vertices
                    self.bucket_polygon.set_xy(self.bucket.getVertices())
                    # Re-update the bucket tip marker
                    self.bucket_tip_plot.set_offsets((self.bucket.tipx, self.bucket.tipy)) 
                    # Set collision flag back to False
                    self.bucket.isCollision = False
            self.iterations_ += 1

            return self.bucket_polygon, self.bucket_tip_plot

    def animate(self):
        # Animate the bucket
        self.ani = FuncAnimation(self.fig, self.update, frames=100, init_func=self.init_animation, interval=50, blit=False)
        plt.show()
        return self.ani
    
    def animate_q(self, Q):
        # Animate the bucket
        self.Q = Q
        self.ani = FuncAnimation(self.fig, self.update_q, frames=200, init_func=self.init_animation, interval=50, blit=False)
        plt.show()
        return self.ani

if __name__ == '__main__':
    # Load Q-table
    with open("q_table.pkl", "rb") as f:
        Q = pickle.load(f)
    # Load pre-defined pointcloud
    pointcloud = np.load("captured_coords.npy")

    simulation = Simulation(pointCloud=pointcloud)
    # pprint.pprint(Q)
    simulation.animate_q(Q)
    print('finished animating?')
