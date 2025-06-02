# Numerical Optimization Programming Assignment 01

This project implements line search minimization algorithms for unconstrained optimization problems.

## Project Structure

```
numOptCluade/
├── src/
│   ├── __init__.py
│   ├── unconstrained_min.py    # Main optimization algorithms
│   └── utils.py                # Plotting utilities
├── tests/
│   ├── __init__.py
│   ├── examples.py             # Test functions
│   └── test_unconstrained_min.py # Unit tests
├── requirements.txt            # Python dependencies
├── run_tests.py               # Test runner script
└── README.md                  # This file
```

## Features

### Optimization Algorithms
- **Gradient Descent**: Steepest descent method
- **Newton's Method**: Second-order method using Hessian information
- **Line Search**: Backtracking line search with Wolfe conditions

### Test Functions
1. **Quadratic 1**: Circular contours - f(x) = x₁² + x₂²
2. **Quadratic 2**: Elliptical contours - f(x) = x₁² + 100x₂²
3. **Quadratic 3**: Rotated elliptical contours 
4. **Rosenbrock**: Non-convex banana function - f(x) = 100(x₂-x₁²)² + (1-x₁)²
5. **Linear**: Linear function - f(x) = 2x₁ + 3x₂
6. **Exponential**: Smooth corner triangle contours

### Visualization
- Contour plots with optimization paths
- Function value convergence plots
- Comparison between Gradient Descent and Newton's method

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run All Tests
```bash
python3 run_tests.py
```

### Run Individual Tests
```bash
cd tests
python3 test_unconstrained_min.py
```

## Algorithm Parameters

- **Objective tolerance**: 1e-12
- **Parameter tolerance**: 1e-8  
- **Maximum iterations**: 100 (10,000 for GD on Rosenbrock)
- **Wolfe constant c1**: 0.01
- **Backtracking parameter**: 0.5

## Initial Points

- Most functions: x₀ = [1, 1]ᵀ
- Rosenbrock: x₀ = [-1, 2]ᵀ

## Output

For each test function, the program generates:
1. Contour plot with optimization paths overlay
2. Function value vs iteration plot
3. Console output with final iteration details

All plots are saved as PNG files in the current directory.

## Implementation Details

The code follows object-oriented design with clean separation of concerns:
- `UnconstrainedMinimizer` class handles the optimization logic
- Wolfe conditions ensure sufficient decrease and curvature conditions
- Robust handling of singular Hessians in Newton's method
- Comprehensive visualization utilities

