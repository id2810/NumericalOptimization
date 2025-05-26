"""
Utility functions for plotting optimization results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Dict, Any


def plot_contours_with_path(f: Callable, x_limits: tuple, y_limits: tuple, 
                           paths: Optional[Dict[str, List[np.ndarray]]] = None,
                           title: str = "Contour Plot", levels: int = 20,
                           figsize: tuple = (10, 8)) -> plt.Figure:
    """
    Plot contour lines of function f with optimization paths.
    
    Args:
        f: Function that takes x and returns (f_val, grad, hess)
        x_limits: (min, max) for x-axis
        y_limits: (min, max) for y-axis  
        paths: Dictionary of {method_name: list_of_points}
        title: Plot title
        levels: Number of contour levels
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grid for contour plot
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Evaluate function on grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            try:
                f_val, _, _ = f(point, False)
                Z[i, j] = f_val
            except:
                Z[i, j] = np.inf
    
    # Handle infinite values
    Z = np.where(np.isfinite(Z), Z, np.nanmax(Z[np.isfinite(Z)]))
    
    # Create contour plot
    if levels == 'auto':
        contour = ax.contour(X, Y, Z, colors='gray', alpha=0.6)
    else:
        # Use logarithmic levels for better visualization
        z_min, z_max = np.nanmin(Z), np.nanmax(Z)
        if z_min <= 0:
            z_min = 1e-10
        log_levels = np.logspace(np.log10(z_min), np.log10(z_max), levels)
        contour = ax.contour(X, Y, Z, levels=log_levels, colors='gray', alpha=0.6)
    
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Plot optimization paths
    if paths:
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, (method_name, path) in enumerate(paths.items()):
            if path:
                path_array = np.array(path)
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                
                # Plot path
                ax.plot(path_array[:, 0], path_array[:, 1], 
                       color=color, linewidth=2, alpha=0.8, label=method_name)
                
                # Mark start and end points
                ax.plot(path_array[0, 0], path_array[0, 1], 
                       marker='o', color=color, markersize=8, markerfacecolor='white',
                       markeredgecolor=color, markeredgewidth=2)
                ax.plot(path_array[-1, 0], path_array[-1, 1], 
                       marker=marker, color=color, markersize=8)
                
                # Add arrows to show direction
                for j in range(0, len(path_array)-1, max(1, len(path_array)//10)):
                    dx = path_array[j+1, 0] - path_array[j, 0]
                    dy = path_array[j+1, 1] - path_array[j, 1]
                    ax.arrow(path_array[j, 0], path_array[j, 1], dx, dy,
                            head_width=0.02*(x_max-x_min), head_length=0.02*(y_max-y_min),
                            fc=color, ec=color, alpha=0.6)
        
        ax.legend()
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    
    plt.tight_layout()
    return fig


def plot_function_values(function_values: Dict[str, List[float]], 
                        title: str = "Function Value vs Iteration",
                        figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Plot function values vs iteration number for different methods.
    
    Args:
        function_values: Dictionary of {method_name: list_of_function_values}
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    linestyles = ['-', '--', '-.', ':', '-']
    
    for i, (method_name, values) in enumerate(function_values.items()):
        if values:
            iterations = range(len(values))
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            
            ax.plot(iterations, values, color=color, linestyle=linestyle, 
                   linewidth=2, marker='o', markersize=4, label=method_name)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Function Value')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')  # Logarithmic scale for better visualization
    
    plt.tight_layout()
    return fig


def get_plot_limits(function_name: str) -> tuple:
    """
    Get appropriate plot limits for different functions.
    
    Returns:
        Tuple of (x_limits, y_limits)
    """
    limits = {
        'quadratic1': ((-2, 2), (-2, 2)),
        'quadratic2': ((-2, 2), (-1, 1)),
        'quadratic3': ((-2, 2), (-2, 2)),
        'rosenbrock': ((-2, 2), (-1, 3)),
        'linear': ((-3, 3), (-3, 3)),
        'exponential': ((-2, 2), (-1, 1))
    }
    
    return limits.get(function_name, ((-3, 3), (-3, 3)))


def save_plots(fig: plt.Figure, filename: str, dpi: int = 300):
    """Save figure to file."""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved as {filename}")


def get_function_title(function_name: str) -> str:
    """Get descriptive title for function."""
    titles = {
        'quadratic1': 'Quadratic Function (Circular Contours): f(x) = x₁² + x₂²',
        'quadratic2': 'Quadratic Function (Elliptical Contours): f(x) = x₁² + 100x₂²', 
        'quadratic3': 'Quadratic Function (Rotated Elliptical Contours)',
        'rosenbrock': 'Rosenbrock Function: f(x) = 100(x₂-x₁²)² + (1-x₁)²',
        'linear': 'Linear Function: f(x) = 2x₁ + 3x₂',
        'exponential': 'Exponential Function: f(x) = eˣ¹⁺³ˣ²⁻⁰·¹ + eˣ¹⁻³ˣ²⁻⁰·¹ + e⁻ˣ¹⁻⁰·¹'
    }
    
    return titles.get(function_name, f'{function_name.title()} Function')
