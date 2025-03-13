from problem import Problem
from search import LocalSearchStrategy

# Create the problem instance
problem = Problem('monalisa.jpg')

# Create the search strategy instance
strategy = LocalSearchStrategy()

# Define a temperature schedule (exponential decay)
def temperature_schedule(t):
    return 100 * 0.95**t  # Initial temp = 100, cooling rate = 0.95

# Run the Simulated Annealing Search algorithm
path = strategy.simulated_annealing_search(problem, temperature_schedule)

# Visualize the result
print(f"Path length: {len(path)}")
print(f"Final state value: {path[-1][2]}")
problem.draw_path(path)