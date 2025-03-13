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
        
    def local_beam_search(self, problem, k):
        """
        Implement the Local Beam Search algorithm.
        
        Args:
            problem: An instance of the Problem class
            k: A positive integer, the maximal number of states maintained
               at each step of the algorithm
            
        Returns:
            list: A list of tuples (x, y, z) representing the path from the initial
                  state to the resulting state.
        """
        # Generate k random initial states
        current_states = []
        for _ in range(k):
            current_states.append(problem.random_state())
        
        # Initialize the best state found
        best_state = max(current_states, key=lambda s: s[2])
        
        # Track paths for all states
        paths = {}
        for state in current_states:
            paths[(state[0], state[1])] = [state]
        
        # Set parameters
        max_iterations = 100
        iteration = 0
        
        while iteration < max_iterations:
            # Generate all neighbors with their paths
            all_successors = []
            
            for state in current_states:
                x, y, _ = state
                current_path = paths[(x, y)]
                
                for neighbor in problem.get_neighbors(x, y):
                    nx, ny, _ = neighbor
                    
                    # Create new path for this neighbor
                    new_path = current_path.copy()
                    new_path.append(neighbor)
                    
                    all_successors.append((neighbor, new_path))
            
            if not all_successors:
                break
            
            # Sort by state value (descending)
            all_successors.sort(key=lambda item: item[0][2], reverse=True)
            
            # Select k best successors
            new_states = []
            seen = set()  # To avoid duplicates
            
            for state, path in all_successors:
                state_key = (state[0], state[1])
                if state_key not in seen and len(new_states) < k:
                    new_states.append(state)
                    paths[state_key] = path
                    seen.add(state_key)
            
            # If no new states, we're done
            if not new_states:
                break
            
            # Update best state if we found better
            new_best = max(new_states, key=lambda s: s[2])
            if new_best[2] > best_state[2]:
                best_state = new_best
            else:
                # No improvement, we're done
                break
            
            current_states = new_states
            iteration += 1
        
        # Return the path to the best state
        return paths[(best_state[0], best_state[1])]