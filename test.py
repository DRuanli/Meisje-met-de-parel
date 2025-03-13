from problem import Problem
from search import LocalSearchStrategy

# Create the problem instance
problem = Problem('monalisa.jpg')

# Create the search strategy instance
strategy = LocalSearchStrategy()

# Run the Local Beam Search algorithm
k = 5  # Number of beams to maintain
path = strategy.local_beam_search(problem, k)

# Visualize the result
print(f"Path length: {len(path)}")
print(f"Final state value: {path[-1][2]}")
problem.draw_path(path)