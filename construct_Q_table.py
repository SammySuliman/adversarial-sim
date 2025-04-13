import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from classes import Obstacle, Goal
import pprint

# Load pre-defined pointcloud
pointcloud = np.load("captured_coords.npy")

# the dimensions of the figure taken from the figure in the Simulation class
fig_height = 800
fig_width = 800
# The scaling factors
scale_x = 20 / fig_width  # Scale factor for x coordinates (from 0-20 range)
scale_y = 20 / fig_height  # Scale factor for y coordinates (from 0-20 range)

# Convert pixel coordinates to normalized mpl coordinates
pointcloud[:, 0] = pointcloud[:, 0] * scale_x
pointcloud[:, 1] = (fig_height - pointcloud[:, 1]) * scale_y  # Flip y-axis to match mpl coordinate system

# We are only interested in the pointcloud within these bounds
x_bounds = (0,20)
y_bounds = (0,20)

obstacle = Obstacle(dimensions=(4, 3), CoM=(8, 8))
goal = Goal(r=2, CoM=(15, 15)) 

vertices_obs = obstacle.getVertices()
# Find min x and min y
min_x = np.min(vertices_obs[:, 0])
min_y = np.min(vertices_obs[:, 1])
# Find max x and max y
max_x = np.max(vertices_obs[:, 0])
max_y = np.max(vertices_obs[:, 1])

def reward_fxn(R):
    for i in range(x_bounds[0], x_bounds[1]+1):
        for j in range(y_bounds[0], y_bounds[1]+1):
            dist_from_goal = np.sqrt((i- goal.CoM[0]) ** 2 + (j - goal.CoM[1]) ** 2)
            distances_from_pointcloud = [np.linalg.norm(point - np.array((i, j))) for point in pointcloud]
            # Find index of minimum distance
            min_index = np.argmin(distances_from_pointcloud)
            closest_point = pointcloud[min_index]  # The corresponding pointpoint
            dist_from_nearest_point = np.sqrt((i- closest_point[0]) ** 2 + (j - closest_point[1]) ** 2)
            # Check if the point exists in the array
            point = np.array([i, j])
            contains_point = np.any(np.all(pointcloud == point, axis=1))
            if dist_from_goal < goal.radius:
                reward = 100
                R[(i,j)] += 100
            else:
                reward = 1 / dist_from_goal
                R[(i,j)] +=  1 / dist_from_goal
            if i >= min_x and i <= max_x and j >= min_y and j <= max_y:
                reward = -100
                R[(i,j)] -= 100
            if contains_point:
                reward = 10
                R[(i,j)] += 10
            else:
                reward = 1 / dist_from_nearest_point
                R[(i,j)] +=  1 / dist_from_nearest_point
    return R

def possible_future_states(current_state):
    (i, j) = current_state
    if i == 0:
        if j == 0:
            possible_actions = ['N', 'E', 'NE']
            possible_states = [(0, 1), (1, 0), (1, 1)]
            return possible_states, possible_actions
        elif j == y_bounds[1]:
            possible_actions = ['S', 'E', 'SE']
            possible_states = [(0, y_bounds[1]-1), (1, y_bounds[1]), (1, y_bounds[1]-1)]
            return possible_states, possible_actions
        else:
            possible_actions = ['S', 'N', 'SE', 'E', 'NE']
            possible_states = [(0, j-1), (0, j+1), (1, j-1), (1, j), (1, j+1)]
            return possible_states, possible_actions
    elif i == x_bounds[1]:
        if j == 0:
            possible_actions = ['N', 'W', 'NW']
            possible_states = [(x_bounds[1], 1), (x_bounds[1]-1, 0), (x_bounds[1]-1, 1)]
            return possible_states, possible_actions
        elif j == y_bounds[1]:
            possible_actions = ['S', 'W', 'SW']
            possible_states = [(x_bounds[1], y_bounds[1]-1), (x_bounds[1]-1, y_bounds[1]), (x_bounds[1]-1, y_bounds[1]-1)]
            return possible_states, possible_actions
        else:
            possible_actions = ['S', 'N', 'SW', 'W', 'NW']
            possible_states = [(x_bounds[1], j-1), (x_bounds[1], j+1), (x_bounds[1]-1, j-1), (x_bounds[1]-1, j), (x_bounds[1]-1, j+1)]
            return possible_states, possible_actions
    elif j == 0:
        possible_actions = ['W', 'E', 'NW', 'N', 'NE']
        possible_states = [(i-1, 0), (i+1, 0), (i-1, 1), (i, 1), (i+1, 1)]
        return possible_states, possible_actions
    elif j == y_bounds[1]:
        possible_actions = ['W', 'E', 'SW', 'S', 'SE']
        possible_states = [(i-1, y_bounds[1]), (i+1, y_bounds[1]), (i-1, y_bounds[1]-1), (i, y_bounds[1]-1), (i+1, y_bounds[1]-1)]
        return possible_states, possible_actions
    else:
        possible_actions = ['SW', 'W', 'NW', 'N', 'NE', 'E', 'SE', 'S']
        possible_states = [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j+1), (i+1, j+1), (i+1, j), (i+1, j-1), (i, j-1)]
        return possible_states, possible_actions
    
def possible_future_states2(current_state):
    (i, j) = current_state
    if i == 0:
        if j == 0:
            possible_actions = ['N', 'E', 'NE']
            possible_states = {(0, 0): possible_actions}
            return possible_states
        elif j == y_bounds[1]:
            possible_actions = ['S', 'E', 'SE']
            possible_states = {(0, y_bounds[1]): possible_actions}
            return possible_states
        else:
            possible_actions = ['S', 'N', 'SE', 'E', 'NE']
            possible_states = {(0, j): possible_actions}
            return possible_states
    elif i == x_bounds[1]:
        if j == 0:
            possible_actions = ['N', 'W', 'NW']
            possible_states = {(x_bounds[1], 0): possible_actions}
            return possible_states
        elif j == y_bounds[1]:
            possible_actions = ['S', 'W', 'SW']
            possible_states = {(x_bounds[1], y_bounds[1]): possible_actions}
            return possible_states
        else:
            possible_actions = ['S', 'N', 'SW', 'W', 'NW']
            possible_states = {(x_bounds[1], j): possible_actions}
            return possible_states
    elif j == 0:
        possible_actions = ['W', 'E', 'NW', 'N', 'NE']
        possible_states = {(i, 0): possible_actions}
        return possible_states
    elif j == y_bounds[1]:
        possible_actions = ['W', 'E', 'SW', 'S', 'SE']
        possible_states = {(i, y_bounds[1]): possible_actions}
        return possible_states
    else:
        possible_actions = ['SW', 'W', 'NW', 'N', 'NE', 'E', 'SE', 'S']
        possible_states = {(i, j): possible_actions}
        return possible_states
            
def bellman_eqn(Q, R, alpha=0.1, gamma=0.95):
    for i in range(x_bounds[0], x_bounds[1]+1):
        for j in range(y_bounds[0], y_bounds[1]+1):
            #print(i, j)
            possible_states, possible_actions = possible_future_states((i, j))
            #print(possible_states)
            #print(possible_actions)
            for index, a in enumerate(possible_actions):
                # print('next action', a)
                next_state = possible_states[index]
                # print('next state', next_state)
                possible_states2, _ = possible_future_states(next_state)
                max_Q = np.max(possible_states2)
                Q[(i,j, a)] = (1 - alpha) * Q[(i,j, a)] + alpha * (R[i, j] + gamma * max_Q - Q[(i,j, a)])
    return Q

def bellman_eqn2(Q, R, starting_state, alpha=0.1, gamma=0.95):
    # We will try an epsilon-greedy approach to filling in Q
    # With probability epsilon
    possible_states, possible_actions = possible_future_states(starting_state)
    for i in range(starting_state[0], x_bounds[1]+1):
        for j in range(starting_state[1], y_bounds[1]+1):
            possible_states, possible_actions = possible_future_states((i, j))
            for index, a in enumerate(possible_actions):
                next_state = possible_states[index]
                possible_states2, _ = possible_future_states(next_state)
                max_Q = np.max(possible_states2)
                Q[(i,j, a)] = (1 - alpha) * Q[(i,j, a)] + alpha * (R[i, j] + gamma * max_Q - Q[(i,j, a)])
    return Q

if __name__ == '__main__':
    R = defaultdict(float)
    Q = defaultdict(lambda: 0.0)
    R = reward_fxn(R)
    #pprint.pprint(R)
    tol = 0.0
    num_iterations = 0
    diff = 1e3
    #starting_state = (3, 3)
    while diff > tol:
        Q2 = bellman_eqn(Q, R)
        Q_arr = np.array(list(Q.values()))
        Q2_arr = np.array(list(Q2.values()))
        diff = np.max(np.abs(Q2_arr - Q_arr))
        #print('diff', diff)
        Q = Q2
        i = np.random.randint(0, 21)
        j = np.random.randint(0, 21)
        #starting_state = (i, j)
        #print(starting_state)
        num_iterations += 1
    pprint.pprint(Q)
    #print('num iterations', num_iterations)
    # Save Q-table
    Q.default_factory = None  # Remove lambda function
    with open("q_table.pkl", "wb") as f:
        pickle.dump(Q, f)

