import tkinter as tk
import numpy as np
from collections import defaultdict
import pickle

def capture_coords():

    # Function to capture drag motion
    def capture_drag(event):
        x, y = event.x, event.y
        path_coordinates.append((x, y))
        canvas.create_oval(x-2, y-2, x+2, y+2, fill="blue", outline="blue")  # Draw on canvas

    # Set up the Tkinter window
    root = tk.Tk()
    root.title("Capture Dragging Motion")
    root.geometry("800x600")

    # Set up canvas
    canvas = tk.Canvas(root, bg="white", width=800, height=600)
    canvas.pack(fill="both", expand=True)

    # List to store coordinates
    path_coordinates = []

    # Bind the drag motion to the canvas
    canvas.bind("<B1-Motion>", capture_drag)

    # Run the Tkinter main loop
    root.mainloop()

    # Print the captured coordinates after closing the window
    # print("Captured Path Coordinates:", path_coordinates)

    return path_coordinates

if __name__ == '__main__':
    capture_array = capture_coords()
    Q = defaultdict(float)
    # Save the defaultdict to a file
    with open('Q.pkl', 'wb') as file:
        pickle.dump(Q, file)
    #np.save("captured_coords.npy", capture_array)
