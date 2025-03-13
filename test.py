#!/usr/bin/env python3
from problem import Problem
from search import LocalSearchStrategy
import time
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
import json


def create_temperature_schedule(initial_temp, cooling_rate):
    """
    Create a temperature schedule function for simulated annealing.
    
    Args:
        initial_temp: Initial temperature
        cooling_rate: Rate at which temperature decreases (between 0 and 1)
        
    Returns:
        function: A temperature schedule function that takes a time step
    """
    return lambda t: initial_temp * (cooling_rate ** t)


def save_results(results, filename):
    """
    Save results to a JSON file.
    
    Args:
        results: Dictionary containing results
        filename: Name of the file to save to
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)


def run_algorithm(problem, strategy, algorithm, params, save_path=False, output_dir=None):
    """
    Run the specified algorithm and return results.
    
    Args:
        problem: Problem instance
        strategy: LocalSearchStrategy instance
        algorithm: Algorithm name to run
        params: Parameters for the algorithm
        save_path: Whether to save the path visualization
        output_dir: Directory to save visualizations
        
    Returns:
        dict: Results including path, execution time, etc.
    """
    print(f"\n=== Running {algorithm} ===")
    start_time = time.time()
    
    if algorithm == "random_restart_hill_climbing":
        path = strategy.random_restart_hill_climbing(problem, params["num_trials"])
    elif algorithm == "simulated_annealing":
        schedule = create_temperature_schedule(
            params["initial_temp"], 
            params["cooling_rate"]
        )
        path = strategy.simulated_annealing_search(problem, schedule)
    elif algorithm == "local_beam_search":
        path = strategy.local_beam_search(problem, params["k"])
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    execution_time = time.time() - start_time
    
    results = {
        "algorithm": algorithm,
        "path_length": len(path),
        "final_value": float(path[-1][2]),  # Convert numpy types to native Python for JSON
        "execution_time": execution_time,
        "start_state": (int(path[0][0]), int(path[0][1]), float(path[0][2])),
        "end_state": (int(path[-1][0]), int(path[-1][1]), float(path[-1][2])),
        "parameters": params
    }
    
    print(f"Path length: {len(path)}")
    print(f"Final state value: {path[-1][2]}")
    print(f"Execution time: {execution_time:.2f} seconds")
    
    if save_path and output_dir:
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d')
        
        # Draw the state space
        ax.plot_surface(problem.X_grid, problem.Y_grid, problem.Z, rstride=1, cstride=1, 
                       cmap='viridis', edgecolor='none', alpha=0.7)
        
        # Extract coordinates from path
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        z_coords = [p[2] for p in path]
        
        # Draw the path
        ax.plot(x_coords, y_coords, z_coords, 'r-', linewidth=2, zorder=10)
        
        # Mark the start and end points
        ax.scatter(x_coords[0], y_coords[0], z_coords[0], color='green', s=100, label='Start')
        ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', s=100, label='End')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Brightness)')
        ax.set_title(f'{algorithm} Path Visualization')
        ax.legend()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"{algorithm}_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Path visualization saved to {filename}")
    else:
        # Display the path visualization
        problem.draw_path(path)
    
    return {"results": results, "path": path}


def compare_results(all_results):
    """
    Compare and display results of all algorithms.
    
    Args:
        all_results: Dictionary containing results for each algorithm
    """
    print("\n=== Algorithm Comparison ===")
    headers = ["Algorithm", "Final Value", "Path Length", "Time (s)"]
    print(f"{headers[0]:<30} | {headers[1]:<12} | {headers[2]:<12} | {headers[3]:<10}")
    print("-" * 70)
    
    for alg_name, result in all_results.items():
        alg_display = alg_name.replace('_', ' ').title()
        print(f"{alg_display:<30} | {result['results']['final_value']:<12.2f} | "
              f"{result['results']['path_length']:<12} | {result['results']['execution_time']:<10.2f}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run local search algorithms on an image-based state space.")
    
    # Image selection
    parser.add_argument("--image", type=str, default="monalisa.jpg",
                        help="Path to the image file for generating the state space")
    
    # Algorithm selection
    parser.add_argument("--algorithms", type=str, nargs="+", 
                        choices=["all", "rrhc", "sa", "lbs"],
                        default=["all"],
                        help="Algorithms to run (rrhc: Random Restart Hill Climbing, "
                             "sa: Simulated Annealing, lbs: Local Beam Search, all: All algorithms)")
    
    # Algorithm parameters
    parser.add_argument("--num-trials", type=int, default=10,
                        help="Number of trials for Random Restart Hill Climbing")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of states for Local Beam Search")
    parser.add_argument("--initial-temp", type=float, default=100.0,
                        help="Initial temperature for Simulated Annealing")
    parser.add_argument("--cooling-rate", type=float, default=0.95,
                        help="Cooling rate for Simulated Annealing")
    
    # Visualization options
    parser.add_argument("--save-visualizations", action="store_true",
                        help="Save visualizations instead of displaying them")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save visualizations and results")
    parser.add_argument("--show-state-space", action="store_true",
                        help="Show the state space visualization")
    parser.add_argument("--save-results", action="store_true",
                        help="Save results to JSON file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.save_visualizations or args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize problem and strategy
    print(f"Loading problem from image: {args.image}")
    try:
        problem = Problem(args.image)
    except Exception as e:
        print(f"Error loading image: {e}")
        print("Please make sure the image file exists and is accessible.")
        return
    
    strategy = LocalSearchStrategy()
    
    # Determine which algorithms to run
    algorithms_to_run = []
    if "all" in args.algorithms or "rrhc" in args.algorithms:
        algorithms_to_run.append("random_restart_hill_climbing")
    if "all" in args.algorithms or "sa" in args.algorithms:
        algorithms_to_run.append("simulated_annealing")
    if "all" in args.algorithms or "lbs" in args.algorithms:
        algorithms_to_run.append("local_beam_search")
    
    # Initialize parameters for each algorithm
    algorithm_params = {
        "random_restart_hill_climbing": {
            "num_trials": args.num_trials
        },
        "simulated_annealing": {
            "initial_temp": args.initial_temp,
            "cooling_rate": args.cooling_rate
        },
        "local_beam_search": {
            "k": args.k
        }
    }
    
    # Run each algorithm and collect results
    all_results = {}
    for algorithm in algorithms_to_run:
        result = run_algorithm(
            problem, 
            strategy, 
            algorithm, 
            algorithm_params[algorithm],
            args.save_visualizations,
            args.output_dir
        )
        all_results[algorithm] = result
    
    # Compare results
    if len(all_results) > 1:
        compare_results(all_results)
    
    # Save results if requested
    if args.save_results:
        result_data = {alg: result["results"] for alg, result in all_results.items()}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(args.output_dir, f"results_{timestamp}.json")
        save_results(result_data, results_file)
        print(f"\nResults saved to {results_file}")
    
    # Show state space visualization if requested
    if args.show_state_space:
        print("\nDisplaying state space visualization...")
        problem.show()


if __name__ == "__main__":
    main()