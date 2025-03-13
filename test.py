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
import logging
import sys
from logging.handlers import RotatingFileHandler


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
    logger = logging.getLogger("LocalSearch")
    
    try:
        with open(filename, 'w') as f:
            # Add metadata to results
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "python_version": sys.version,
            }
            
            enhanced_results = {
                "metadata": metadata,
                "results": results
            }
            
            json.dump(enhanced_results, f, indent=4)
        logger.debug(f"Results successfully saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save results to {filename}: {e}")
        logger.exception("Exception details:")


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
    logger = logging.getLogger("LocalSearch")
    
    logger.info(f"Running algorithm: {algorithm} with parameters: {params}")
    print(f"\n=== Running {algorithm} ===")
    
    # Record start time for performance measurement
    start_time = time.time()
    
    try:
        # Run the appropriate algorithm
        if algorithm == "random_restart_hill_climbing":
            logger.info(f"Starting Random Restart Hill-Climbing with {params['num_trials']} trials")
            path = strategy.random_restart_hill_climbing(problem, params["num_trials"])
        elif algorithm == "simulated_annealing":
            schedule = create_temperature_schedule(
                params["initial_temp"], 
                params["cooling_rate"]
            )
            logger.info(f"Starting Simulated Annealing with initial temp {params['initial_temp']} and cooling rate {params['cooling_rate']}")
            path = strategy.simulated_annealing_search(problem, schedule)
        elif algorithm == "local_beam_search":
            logger.info(f"Starting Local Beam Search with k={params['k']}")
            path = strategy.local_beam_search(problem, params["k"])
        else:
            error_msg = f"Unknown algorithm: {algorithm}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Log the results
        logger.info(f"Algorithm completed in {execution_time:.2f} seconds")
        logger.info(f"Path length: {len(path)}")
        logger.info(f"Start state: ({path[0][0]}, {path[0][1]}) with value {path[0][2]}")
        logger.info(f"End state: ({path[-1][0]}, {path[-1][1]}) with value {path[-1][2]}")
        
        if len(path) > 1:
            # Calculate improvement
            improvement = path[-1][2] - path[0][2]
            improvement_percent = (improvement / path[0][2]) * 100 if path[0][2] != 0 else float('inf')
            logger.info(f"Value improvement: {improvement:.2f} ({improvement_percent:.2f}%)")
            
    except Exception as e:
        logger.error(f"Error executing {algorithm}: {e}")
        logger.exception("Exception details:")
        print(f"Error executing {algorithm}: {e}")
        # Return empty results in case of error
        return {"results": {"algorithm": algorithm, "error": str(e)}, "path": []}
    
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


def setup_logger(log_level, log_file=None, console_output=True):
    """
    Set up the logger with appropriate handlers and formatting.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, no file logging)
        console_output: Whether to output logs to console
        
    Returns:
        Logger object
    """
    # Create logger
    logger = logging.getLogger("LocalSearch")
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add handlers
    handlers = []
    
    # File handler (if specified)
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # Use rotating file handler to prevent huge log files
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Console handler (if specified)
    if console_output:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # Clear existing handlers and add new ones
    logger.handlers = []
    for handler in handlers:
        logger.addHandler(handler)
    
    return logger


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
                        
    # Logging options
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO", help="Set the logging level")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to log file (default: output_dir/local_search.log)")
    parser.add_argument("--no-console-log", action="store_true",
                        help="Disable logging to console")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.save_visualizations or args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        
    # Set up logging
    log_level = getattr(logging, args.log_level)
    if args.log_file is None and (args.save_results or args.save_visualizations):
        args.log_file = os.path.join(args.output_dir, "local_search.log")
    
    logger = setup_logger(
        log_level=log_level,
        log_file=args.log_file,
        console_output=not args.no_console_log
    )
    
    logger.info("=" * 80)
    logger.info("Starting local search algorithm test")
    logger.info(f"Command-line arguments: {vars(args)}")
    logger.info("=" * 80)
    
    # Initialize problem and strategy
    logger.info(f"Loading problem from image: {args.image}")
    print(f"Loading problem from image: {args.image}")
    try:
        problem = Problem(args.image)
        logger.info(f"Problem loaded successfully. State space dimensions: {problem.width}x{problem.height}")
    except Exception as e:
        error_msg = f"Error loading image: {e}"
        logger.error(error_msg)
        logger.exception("Exception details:")
        print(error_msg)
        print("Please make sure the image file exists and is accessible.")
        return
    
    strategy = LocalSearchStrategy()
    logger.info("LocalSearchStrategy initialized")
    
    # Determine which algorithms to run
    algorithms_to_run = []
    if "all" in args.algorithms or "rrhc" in args.algorithms:
        algorithms_to_run.append("random_restart_hill_climbing")
    if "all" in args.algorithms or "sa" in args.algorithms:
        algorithms_to_run.append("simulated_annealing")
    if "all" in args.algorithms or "lbs" in args.algorithms:
        algorithms_to_run.append("local_beam_search")
    
    logger.info(f"Algorithms to run: {algorithms_to_run}")
    
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
    
    logger.debug(f"Algorithm parameters: {algorithm_params}")
    
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
        logger.info(f"Results saved to {results_file}")
        print(f"\nResults saved to {results_file}")
        
        # Save algorithm run history
        history_file = os.path.join(args.output_dir, "search_history.log")
        try:
            with open(history_file, 'a') as f:
                f.write(f"Run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Image: {args.image}\n")
                for alg, result in all_results.items():
                    if "error" in result["results"]:
                        f.write(f"{alg}: ERROR - {result['results']['error']}\n")
                    else:
                        f.write(f"{alg}: Final value = {result['results']['final_value']}, "
                               f"Path length = {result['results']['path_length']}, "
                               f"Time = {result['results']['execution_time']:.2f}s\n")
                f.write("-" * 80 + "\n")
            logger.info(f"Run history appended to {history_file}")
        except Exception as e:
            logger.error(f"Failed to write to history file: {e}")
    
    # Show state space visualization if requested
    if args.show_state_space:
        logger.info("Displaying state space visualization")
        print("\nDisplaying state space visualization...")
        problem.show()
        
    logger.info("Test script execution completed successfully")
    print("\nTest completed. Check log file for detailed execution information.")


if __name__ == "__main__":
    main()