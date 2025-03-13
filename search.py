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
        
    def simulated_annealing_search(self, problem, schedule):
        """
        Implement the Simulated Annealing Search algorithm.
        
        Args:
            problem: An instance of the Problem class
            schedule: A function taking a time step t and returning a temperature value
            
        Returns:
            list: A list of tuples (x, y, z) representing the path from the initial
                  state to the resulting state.
        """
        # Get a random starting state
        current_x, current_y, current_z = problem.random_state()
        
        # Initialize path
        path = [(current_x, current_y, current_z)]
        
        # Set initial time step
        t = 1
        max_steps = 1000  # Prevent infinite loops
        
        # Continue until temperature is very low or max steps reached
        while t < max_steps:
            # Get current temperature
            T = schedule(t)
            
            # Check termination condition
            if T < 0.1:  # Very low temperature
                break
            
            # Get a random neighbor
            neighbors = problem.get_neighbors(current_x, current_y)
            if not neighbors:
                break
                
            next_x, next_y, next_z = neighbors[np.random.randint(0, len(neighbors))]
            
            # Compute the difference in value (we want to maximize z)
            delta_e = next_z - current_z
            
            # Decide whether to accept the neighbor
            accept_move = False
            if delta_e > 0:  # Better state, always accept
                accept_move = True
            else:
                # Accept with probability e^(delta_e / T)
                p = np.exp(delta_e / T)
                if np.random.random() < p:
                    accept_move = True
            
            # Update current state if move is accepted
            if accept_move:
                current_x, current_y, current_z = next_x, next_y, next_z
                path.append((current_x, current_y, current_z))
            
            # Increment time step
            t += 1
        
        return path