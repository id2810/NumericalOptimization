"""
Unconstrained minimization algorithms with line search.
Supports Gradient Descent and Newton methods with Wolfe conditions.
"""

import numpy as np
from typing import Callable, Tuple, List, Optional


class UnconstrainedMinimizer:
    """
    Class for unconstrained minimization using line search methods.
    Supports Gradient Descent and Newton's method.
    """
    
    def __init__(self):
        self.path_x = []  # Store x values for each iteration
        self.path_f = []  # Store function values for each iteration
        self.method_used = None
    
    def minimize(self, f: Callable, x0: np.ndarray, obj_tol: float = 1e-12, 
                param_tol: float = 1e-8, max_iter: int = 100, 
                method: str = 'gd') -> Tuple[np.ndarray, float, bool]:
        """
        Minimize function f starting from x0.
        
        Args:
            f: Objective function that returns (f_val, grad, hess)
            x0: Starting point
            obj_tol: Tolerance for objective function change
            param_tol: Tolerance for parameter change
            max_iter: Maximum number of iterations
            method: 'gd' for gradient descent, 'newton' for Newton's method
            
        Returns:
            Tuple of (final_x, final_f, success_flag)
        """
        self.path_x = []
        self.path_f = []
        self.method_used = method
        
        x = np.array(x0, dtype=float)
        
        # Wolfe conditions parameters
        c1 = 0.01  # Armijo parameter
        c2 = 0.9   # Curvature parameter  
        alpha_init = 1.0
        rho = 0.5  # Backtracking parameter
        
        for i in range(max_iter):
            # Evaluate function, gradient, and Hessian (if needed)
            need_hessian = (method.lower() == 'newton')
            f_val, grad, hess = f(x, need_hessian)
            
            # Store current point
            self.path_x.append(x.copy())
            self.path_f.append(f_val)
            
            # Print iteration info
            print(f"Iteration {i}: x = [{x[0]:.6f}, {x[1]:.6f}], f(x) = {f_val:.10f}")
            
            # Check gradient stopping criterion
            grad_norm = np.linalg.norm(grad)
            if grad_norm < np.sqrt(obj_tol):
                print(f"Converged due to small gradient norm: {grad_norm}")
                return x, f_val, True
            
            # Compute search direction
            if method.lower() == 'gd':
                p = -grad  # Gradient descent direction
            elif method.lower() == 'newton':
                try:
                    # Newton direction: solve H * p = -g
                    p = -np.linalg.solve(hess, grad)
                    # Check if direction is descent direction
                    if np.dot(grad, p) >= 0:
                        print("Newton direction is not descent, falling back to gradient descent")
                        p = -grad
                except np.linalg.LinAlgError:
                    print("Hessian is singular, falling back to gradient descent")
                    p = -grad
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Line search with Wolfe conditions
            alpha = self._wolfe_line_search(f, x, p, f_val, grad, c1, c2, alpha_init, rho)
            
            if alpha is None:
                print("Line search failed")
                return x, f_val, False
            
            # Update x
            x_new = x + alpha * p
            
            # Check parameter tolerance
            if np.linalg.norm(x_new - x) < param_tol:
                print(f"Converged due to small parameter change: {np.linalg.norm(x_new - x)}")
                self.path_x.append(x_new.copy())
                f_new, _, _ = f(x_new, False)
                self.path_f.append(f_new)
                return x_new, f_new, True
            
            # Check objective tolerance
            f_new, _, _ = f(x_new, False)
            if abs(f_new - f_val) < obj_tol:
                print(f"Converged due to small objective change: {abs(f_new - f_val)}")
                self.path_x.append(x_new.copy())
                self.path_f.append(f_new)
                return x_new, f_new, True
            
            # Newton decrement check for Newton's method
            if method.lower() == 'newton':
                try:
                    newton_decrement = 0.5 * grad.T @ np.linalg.solve(hess, grad)
                    if newton_decrement < obj_tol:
                        print(f"Converged due to small Newton decrement: {newton_decrement}")
                        self.path_x.append(x_new.copy())
                        self.path_f.append(f_new)
                        return x_new, f_new, True
                except np.linalg.LinAlgError:
                    pass  # Skip Newton decrement check if Hessian is singular
            
            x = x_new
        
        print(f"Maximum iterations ({max_iter}) reached")
        return x, self.path_f[-1] if self.path_f else f_val, False
    
    def _wolfe_line_search(self, f: Callable, x: np.ndarray, p: np.ndarray, 
                          f0: float, grad0: np.ndarray, c1: float, c2: float,
                          alpha_init: float, rho: float) -> Optional[float]:
        """
        Backtracking line search satisfying Wolfe conditions.
        """
        alpha = alpha_init
        
        for _ in range(50):  # Maximum line search iterations
            x_new = x + alpha * p
            f_new, grad_new, _ = f(x_new, False)
            
            # Armijo condition
            if f_new <= f0 + c1 * alpha * np.dot(grad0, p):
                # Curvature condition (weak Wolfe)
                if np.dot(grad_new, p) >= c2 * np.dot(grad0, p):
                    return alpha
            
            # Backtrack
            alpha *= rho
            
            if alpha < 1e-16:
                break
        
        # If Wolfe conditions not satisfied, return step that satisfies Armijo
        alpha = alpha_init
        for _ in range(50):
            x_new = x + alpha * p
            f_new, _, _ = f(x_new, False)
            
            if f_new <= f0 + c1 * alpha * np.dot(grad0, p):
                return alpha
                
            alpha *= rho
            
            if alpha < 1e-16:
                break
                
        return None
    
    def get_path(self) -> Tuple[List[np.ndarray], List[float]]:
        """Return the optimization path."""
        return self.path_x, self.path_f


# Convenience function interface
def minimize_function(f: Callable, x0: np.ndarray, obj_tol: float = 1e-12,
                     param_tol: float = 1e-8, max_iter: int = 100,
                     method: str = 'gd') -> Tuple[np.ndarray, float, bool, List[np.ndarray], List[float]]:
    """
    Minimize function f starting from x0.
    
    Returns:
        Tuple of (final_x, final_f, success_flag, path_x, path_f)
    """
    minimizer = UnconstrainedMinimizer()
    final_x, final_f, success = minimizer.minimize(f, x0, obj_tol, param_tol, max_iter, method)
    path_x, path_f = minimizer.get_path()
    return final_x, final_f, success, path_x, path_f
