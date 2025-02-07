import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from classes import Obstacle, Goal
import pprint

# Load pre-defined pointcloud
pointcloud = np.load("captured_coords.npy")

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
            if dist_from_goal < goal.radius:
                reward = 100
                R[(i,j)] += 100
            else:
                reward = 1 / dist_from_goal
                R[(i,j)] +=  1 / dist_from_goal
            if i >= min_x and i <= max_x and j >= min_y and j <= max_y:
                reward = -100
                R[(i,j)] -= 100
            if (i,j) in pointcloud:
                reward = 10
                R[(i,j)] += 10
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

if __name__ == '__main__':
    R = defaultdict(float)
    Q = defaultdict(lambda: 0.0)
    R = reward_fxn(R)
    tol = 1e-3
    num_iterations = 0
    diff = 1e3
    while diff > tol:
        print('iter#', num_iterations)
        Q2 = bellman_eqn(Q, R)
        Q_arr = np.array(list(Q.values()))
        Q2_arr = np.array(list(Q2.values()))
        diff = np.max(np.abs(Q2_arr - Q_arr))
        print('diff', diff)
        Q = Q2
        num_iterations += 1
    #pprint.pprint(Q)
    # Save Q-table
    Q.default_factory = None  # Remove lambda function
    with open("q_table.pkl", "wb") as f:
        pickle.dump(Q, f)

