from problem import Problem
from search import LocalSearchStrategy

# Create the problem instance
problem = Problem('monalisa.jpg')

# Create the search strategy instance
strategy = LocalSearchStrategy()

# Run the Random Restart Hill-Climbing algorithm
num_trials = 10  # Number of trials
path = strategy.random_restart_hill_climbing(problem, num_trials)

# Visualize the result
print(f"Path length: {len(path)}")
print(f"Final state value: {path[-1][2]}")
problem.draw_path(path)

# Uncomment to show the state space without a path
problem.show()