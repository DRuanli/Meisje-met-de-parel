from problem import Problem
from search import LocalSearchStrategy
import time

def main():
    print("Loading problem...")
    problem = Problem('Meisje met de parel.jpg')
    strategy = LocalSearchStrategy()
    
    # Define parameters for each algorithm
    num_trials = 10  # For random restart hill climbing
    k = 5  # For local beam search
    
    # Define temperature schedule for simulated annealing
    def temperature_schedule(t):
        return 100 * 0.95**t  # Initial temp = 100, cooling rate = 0.95
    
    # Test Random Restart Hill-Climbing
    print("\n=== Random Restart Hill-Climbing ===")
    start_time = time.time()
    rrhc_path = strategy.random_restart_hill_climbing(problem, num_trials)
    rrhc_time = time.time() - start_time
    
    print(f"Path length: {len(rrhc_path)}")
    print(f"Final state value: {rrhc_path[-1][2]}")
    print(f"Execution time: {rrhc_time:.2f} seconds")
    print("Displaying path visualization...")
    problem.draw_path(rrhc_path)
    
    # Test Simulated Annealing
    print("\n=== Simulated Annealing ===")
    start_time = time.time()
    sa_path = strategy.simulated_annealing_search(problem, temperature_schedule)
    sa_time = time.time() - start_time
    
    print(f"Path length: {len(sa_path)}")
    print(f"Final state value: {sa_path[-1][2]}")
    print(f"Execution time: {sa_time:.2f} seconds")
    print("Displaying path visualization...")
    problem.draw_path(sa_path)
    
    # Test Local Beam Search
    print("\n=== Local Beam Search ===")
    start_time = time.time()
    lbs_path = strategy.local_beam_search(problem, k)
    lbs_time = time.time() - start_time
    
    print(f"Path length: {len(lbs_path)}")
    print(f"Final state value: {lbs_path[-1][2]}")
    print(f"Execution time: {lbs_time:.2f} seconds")
    print("Displaying path visualization...")
    problem.draw_path(lbs_path)
    
    # Compare results
    print("\n=== Algorithm Comparison ===")
    print(f"{'Algorithm':<25} | {'Final Value':<12} | {'Path Length':<12} | {'Time (s)':<10}")
    print("-" * 65)
    print(f"{'Random Restart Hill-Climbing':<25} | {rrhc_path[-1][2]:<12} | {len(rrhc_path):<12} | {rrhc_time:.2f}")
    print(f"{'Simulated Annealing':<25} | {sa_path[-1][2]:<12} | {len(sa_path):<12} | {sa_time:.2f}")
    print(f"{'Local Beam Search':<25} | {lbs_path[-1][2]:<12} | {len(lbs_path):<12} | {lbs_time:.2f}")
    
    # Show state space visualization
    print("\nDisplaying state space visualization...")
    problem.show()

if __name__ == "__main__":
    main()