import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

class Problem:
    def __init__(self, image_filename):
        """
        Initialize the problem with the given image file.
        
        Args:
            image_filename (str): Path to the image file.
        """
        self.X, self.Y, self.Z = self.load_state_space(image_filename)
        self.X_grid, self.Y_grid = np.meshgrid(self.X, self.Y)
        self.height, self.width = self.Z.shape
    
    def load_state_space(self, filename):
        """
        Load and process the image to create the state space.
        
        Args:
            filename (str): Path to the image file.
            
        Returns:
            tuple: X, Y coordinate arrays and Z values (brightness).
        """
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        h, w = img.shape
        X = np.arange(w)
        Y = np.arange(h)
        Z = img
        return X, Y, Z
    
    def get_value(self, x, y):
        """
        Get the evaluation value (z) at the given coordinates.
        
        Args:
            x (int): X coordinate.
            y (int): Y coordinate.
            
        Returns:
            int: The brightness value (z) at the given coordinates.
        """
        if self.is_valid_state(x, y):
            return self.Z[y, x]
        return 0
    
    def is_valid_state(self, x, y):
        """
        Check if the given coordinates are within the state space bounds.
        
        Args:
            x (int): X coordinate.
            y (int): Y coordinate.
            
        Returns:
            bool: True if the coordinates are valid, False otherwise.
        """
        return 0 <= x < self.width and 0 <= y < self.height
    
    def random_state(self):
        """
        Generate a random valid state within the state space.
        
        Returns:
            tuple: A tuple (x, y, z) representing a random state.
        """
        x = np.random.randint(0, self.width)
        y = np.random.randint(0, self.height)
        z = self.get_value(x, y)
        return (x, y, z)
    
    def get_neighbors(self, x, y):
        """
        Get all valid neighboring states of the given coordinates.
        
        Args:
            x (int): X coordinate.
            y (int): Y coordinate.
            
        Returns:
            list: A list of tuples (x, y, z) representing valid neighboring states.
        """
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self.is_valid_state(nx, ny):
                z = self.get_value(nx, ny)
                neighbors.append((nx, ny, z))
        return neighbors
    
    def show(self):
        """
        Visualize the state space as a 3D surface.
        """
        fig = plt.figure(figsize=(8, 6))
        ax = plt.axes(projection='3d')
        ax.plot_surface(self.X_grid, self.Y_grid, self.Z, rstride=1, cstride=1, 
                        cmap='viridis', edgecolor='none')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Brightness)')
        ax.set_title('State Space Visualization')
        plt.show()
    
    def draw_path(self, path):
        """
        Draw the given path on the state space.
        
        Args:
            path (list): A list of tuples (x, y, z) representing the path.
        """
        if not path:
            print("No path to draw!")
            return
        
        # Extract x, y, z coordinates from the path
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        z_coords = [p[2] for p in path]
        
        fig = plt.figure(figsize=(8, 6))
        ax = plt.axes(projection='3d')
        
        # Draw the state space
        ax.plot_surface(self.X_grid, self.Y_grid, self.Z, rstride=1, cstride=1, 
                       cmap='viridis', edgecolor='none', alpha=0.7)
        
        # Draw the path
        ax.plot(x_coords, y_coords, z_coords, 'r-', linewidth=2, zorder=10)
        
        # Mark the start and end points
        ax.scatter(x_coords[0], y_coords[0], z_coords[0], color='green', s=100, label='Start')
        ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', s=100, label='End')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Brightness)')
        ax.set_title('Path Visualization')
        ax.legend()
        plt.show()