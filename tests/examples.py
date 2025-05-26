"""
Objective functions for numerical optimization testing.
All functions return (f, g, h) where:
- f: function value at x
- g: gradient at x  
- h: Hessian at x (only if hessian_flag is True)
"""

import numpy as np


def quadratic1(x, hessian_flag=False):
    """
    Quadratic function f(x) = x^T Q x where Q = [[1, 0], [0, 1]]
    Contour lines are circles.
    """
    x = np.array(x)
    Q = np.array([[1, 0], [0, 1]])
    
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q if hessian_flag else None
    
    return f, g, h


def quadratic2(x, hessian_flag=False):
    """
    Quadratic function f(x) = x^T Q x where Q = [[1, 0], [0, 100]]
    Contour lines are axis-aligned ellipses.
    """
    x = np.array(x)
    Q = np.array([[1, 0], [0, 100]])
    
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q if hessian_flag else None
    
    return f, g, h


def quadratic3(x, hessian_flag=False):
    """
    Quadratic function f(x) = x^T Q x where Q is rotated
    Q = R^T * diag([100, 1]) * R where R is rotation matrix
    Contour lines are rotated ellipses.
    """
    x = np.array(x)
    
    # Rotation matrix R with angle such that R = [[sqrt(3)/2, -0.5], [0.5, sqrt(3)/2]]
    R = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    D = np.array([[100, 0], [0, 1]])
    Q = R.T @ D @ R
    
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q if hessian_flag else None
    
    return f, g, h


def rosenbrock(x, hessian_flag=False):
    """
    Rosenbrock function: f(x) = 100(x2 - x1^2)^2 + (1 - x1)^2
    Famous non-convex optimization benchmark.
    """
    x = np.array(x)
    x1, x2 = x[0], x[1]
    
    f = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    
    # Gradient
    df_dx1 = -400 * x1 * (x2 - x1**2) - 2 * (1 - x1)
    df_dx2 = 200 * (x2 - x1**2)
    g = np.array([df_dx1, df_dx2])
    
    # Hessian
    h = None
    if hessian_flag:
        d2f_dx1dx1 = -400 * (x2 - 3 * x1**2) + 2
        d2f_dx1dx2 = -400 * x1
        d2f_dx2dx1 = -400 * x1
        d2f_dx2dx2 = 200
        h = np.array([[d2f_dx1dx1, d2f_dx1dx2], 
                      [d2f_dx2dx1, d2f_dx2dx2]])
    
    return f, g, h


def linear_function(x, hessian_flag=False):
    """
    Linear function f(x) = a^T x where a = [2, 3]
    Contour lines are straight lines.
    """
    x = np.array(x)
    a = np.array([2, 3])
    
    f = a.T @ x
    g = a
    h = np.zeros((2, 2)) if hessian_flag else None
    
    return f, g, h


def exponential_function(x, hessian_flag=False):
    """
    Function f(x1, x2) = exp(x1 + 3*x2 - 0.1) + exp(x1 - 3*x2 - 0.1) + exp(-x1 - 0.1)
    Contour lines look like smoothed corner triangles.
    """
    x = np.array(x)
    x1, x2 = x[0], x[1]
    
    term1 = np.exp(x1 + 3*x2 - 0.1)
    term2 = np.exp(x1 - 3*x2 - 0.1)
    term3 = np.exp(-x1 - 0.1)
    
    f = term1 + term2 + term3
    
    # Gradient
    df_dx1 = term1 + term2 - term3
    df_dx2 = 3*term1 - 3*term2
    g = np.array([df_dx1, df_dx2])
    
    # Hessian
    h = None
    if hessian_flag:
        d2f_dx1dx1 = term1 + term2 + term3
        d2f_dx1dx2 = 3*term1 - 3*term2
        d2f_dx2dx1 = 3*term1 - 3*term2
        d2f_dx2dx2 = 9*term1 + 9*term2
        h = np.array([[d2f_dx1dx1, d2f_dx1dx2], 
                      [d2f_dx2dx1, d2f_dx2dx2]])
    
    return f, g, h


# Dictionary mapping function names to functions for easy access
FUNCTIONS = {
    'quadratic1': quadratic1,
    'quadratic2': quadratic2, 
    'quadratic3': quadratic3,
    'rosenbrock': rosenbrock,
    'linear': linear_function,
    'exponential': exponential_function
}
