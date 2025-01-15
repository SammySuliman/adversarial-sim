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

matplotlib.use('TkAgg')

class Simulation:
    def __init__(self):
        # Create some sample objects
        self.bucket = Bucket(initial_tip=(3, 3))
        self.obstacle = Obstacle(dimensions=(4, 3), CoM=(8, 8))
        self.goal = Goal(r=2, CoM=(15, 15))  
        self.iterations = 0

        self.pointCloud = np.array(capture_coords())

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

    def update(self, frame, max_iterations=50):
        while self.iterations < max_iterations:
            # Set the current direction for each update
            directions = ['N', 'S', 'W', 'E', 'NW', 'NE', 'SW', 'SE']
            self.current_dir = random.choice(directions)
            # Remove existing bucket from plot
            # self.bucket_polygon.remove()
            # Remove bucket tip marker
            # self.bucket_tip_plot.remove()
            # print('current direction of movement', self.current_dir)
            # Move bucket 1 step in existing direction
            self.bucket.move(velocity=1.0, time=1.0, current_dir=self.current_dir)
            # Update the polygon vertices
            # print('new vertices', self.bucket.getVertices())
            # print('point cloud', self.pointCloud)
            # print('bucket tip', (simulation.bucket.tipx, simulation.bucket.tipy))
            self.bucket_polygon.set_xy(self.bucket.getVertices())
            # Update the bucket tip marker
            self.bucket_tip_plot.set_offsets((self.bucket.tipx, self.bucket.tipy))
            inside_points = self.bucket.gatherMaterial(self.pointCloud)
            # print('gathered material', self.bucket.gathered_material)
            if len(inside_points) != 0:
                # Remove all elements in the sublist
                #self.pointCloud = [item for item in list(self.pointCloud) if item not in self.bucket.gathered_material]
                #self.pointCloud = [point for point in self.pointCloud.tolist() if point not in self.bucket.gathered_material]
                self.pointCloud = np.array([point for point in self.pointCloud.tolist() if not np.any(np.all(point == inside_points, axis=1))])
                # print('new point cloud', self.pointCloud)
                self.scatter.set_offsets(self.pointCloud)


            self.iterations += 1
            print('iter number', self.iterations)

            return self.bucket_polygon, self.bucket_tip_plot
    
    def animate(self):
        # Animate the bucket
        ani = FuncAnimation(self.fig, self.update, frames=50, init_func=self.init_animation, interval=50, blit=False)
        plt.show()

        return ani

if __name__ == '__main__':
    simulation = Simulation()
    simulation.animate()
    print('finished animating?')


    
