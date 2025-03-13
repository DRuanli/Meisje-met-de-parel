# Local Search Algorithms for Optimization

## Project Overview
This project implements and visualizes three local search algorithms to find optimal points on a 3D surface generated from an image. The algorithms navigate the state space seeking states with higher values (higher brightness in the image).

## Algorithms Implemented
1. **Random Restart Hill Climbing**: Performs multiple hill climbing searches from random starting points and returns the best path found.
2. **Simulated Annealing**: Uses a temperature schedule to control the probability of accepting worse states during the search.
3. **Local Beam Search**: Maintains multiple states simultaneously and expands the most promising ones.

## Project Structure
- `monalisa.jpg`: Source image used to generate the state space.
- `problem.py`: Contains the `Problem` class that defines the state space and provides visualization methods.
- `search.py`: Implements the `LocalSearchStrategy` class with all three search algorithms.
- `test.py`: Provides a test harness to run and compare the algorithms.
- `viz3d.py`: Helper script with examples for visualizing the state space.

## Requirements
- Python 3.7+
- NumPy
- Matplotlib
- OpenCV (cv2)

## Installation
1. Ensure Python 3.7 or higher is installed on your system.
2. Install the required packages:
   ```
   pip install numpy matplotlib opencv-python
   ```

## Usage
Run the main test script to execute all three algorithms and view the visualizations:
```
python test.py
```

This will:
1. Load the problem state space from `monalisa.jpg`
2. Run each algorithm and display their execution paths
3. Compare the performance of all three algorithms
4. Show a visualization of the complete state space

## Parameters
You can modify the following parameters in `test.py`:
- `num_trials`: Number of restarts for Random Restart Hill Climbing
- `k`: Number of states maintained in Local Beam Search
- `temperature_schedule`: The cooling schedule for Simulated Annealing

## Visualization
The project provides two main types of visualizations:
1. **State Space Visualization**: A 3D surface representing the entire state space, where the height (Z-axis) corresponds to the pixel brightness in the source image.
2. **Path Visualization**: Shows the path taken by each algorithm from the starting point to the final state.

## Implementation Details

### Problem Formulation
Each state in the state space is represented by:
- `x`: X-coordinate in the image
- `y`: Y-coordinate in the image
- `z`: Pixel brightness at (x,y), representing the value to be maximized

### Random Restart Hill Climbing
This algorithm:
- Starts with a random state
- Repeatedly moves to the best neighboring state until reaching a local maximum
- Restarts from a new random state and keeps track of the best solution found

### Simulated Annealing
This algorithm:
- Starts with a random state and high temperature
- Considers random neighboring states
- Always accepts better states
- Probabilistically accepts worse states based on the current temperature
- Gradually decreases temperature according to the schedule

### Local Beam Search
This algorithm:
- Starts with k random states
- Generates all neighbors of all current states
- Selects the k best states from all generated neighbors
- Returns the path to the best state found

## Results Analysis
When you run the test script, a comparison table is displayed showing:
- Final state value (z) reached by each algorithm
- Path length (number of steps taken)
- Execution time

This allows for easy comparison of the algorithms' performance in terms of solution quality and efficiency.

## License
This project is available under the MIT License.

## References
- Russell, S. J., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.). Pearson.