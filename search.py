import numpy as np

class LocalSearchStrategy:
    def random_restart_hill_climbing(self, problem, num_trial):
        """
        Implement the Random Restart Hill-Climbing algorithm.
        
        Args:
            problem: An instance of the Problem class
            num_trial: Number of trials to perform
            
        Returns:
            list: A list of tuples (x, y, z) representing the path from the initial
                 state to the resulting state.
        """
        best_path = []
        best_value = -1
        
        for _ in range(num_trial):
            # Get a random starting state
            x, y, z = problem.random_state()
            
            # Perform hill climbing from this state
            current_path = self._hill_climbing(problem, x, y)
            
            # Check if this path led to a better solution
            if current_path and current_path[-1][2] > best_value:
                best_path = current_path
                best_value = current_path[-1][2]
                
        return best_path
    
    def _hill_climbing(self, problem, start_x, start_y):
        """
        Helper method to perform standard hill-climbing from a given start state.
        
        Args:
            problem: An instance of the Problem class
            start_x: X coordinate of the starting state
            start_y: Y coordinate of the starting state
            
        Returns:
            list: A list of tuples (x, y, z) representing the path from the initial
                 state to the local maximum.
        """
        # Initialize current state and path
        current_x, current_y = start_x, start_y
        current_z = problem.get_value(current_x, current_y)
        path = [(current_x, current_y, current_z)]
        
        while True:
            # Get all neighbors
            neighbors = problem.get_neighbors(current_x, current_y)
            
            # Find the best neighbor
            best_neighbor = None
            best_neighbor_value = current_z
            
            for nx, ny, nz in neighbors:
                if nz > best_neighbor_value:
                    best_neighbor = (nx, ny, nz)
                    best_neighbor_value = nz
            
            # If no better neighbor, we've reached a local maximum
            if best_neighbor is None:
                break
                
            # Move to the best neighbor
            current_x, current_y, current_z = best_neighbor
            path.append((current_x, current_y, current_z))
            
        return path