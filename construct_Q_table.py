import numpy as np
import pickle

from main import Simulation

initial_tip = (3, 3)
with open('Q.pkl', 'rb') as file:
    Q = pickle.load(file)

for i in range(50):
    pointcloud = np.load("captured_coords.npy")
    simulation = Simulation(pointCloud=pointcloud, initial_tip=initial_tip)
    simulation.animate()
    print('finished animating?')
    for key, value in simulation.Q.items():
        #print('key', key)
        #print('val', value)
        Q[key] += value
    #print('newest Q iteration: ', simulation.Q)
    print('Q', Q)
    initial_tip = (simulation.bucket.tipx, simulation.bucket.tipy)
        
# Save the defaultdict to a file
with open('Q.pkl', 'wb') as file:
    pickle.dump(Q, file)