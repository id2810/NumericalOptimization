"""
Test module for unconstrained minimization algorithms.
Tests both Gradient Descent and Newton's method on various functions.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from unconstrained_min import UnconstrainedMinimizer
from utils import plot_contours_with_path, plot_function_values, get_plot_limits, get_function_title
from tests.examples import FUNCTIONS


class TestUnconstrainedMinimization(unittest.TestCase):
    """Test class for unconstrained minimization algorithms."""
    
    def setUp(self):
        """Set up test parameters."""
        self.obj_tol = 1e-12
        self.param_tol = 1e-8
        self.max_iter_default = 100
        self.max_iter_rosenbrock_gd = 10000
        
        # Initial points
        self.x0_default = np.array([1.0, 1.0])
        self.x0_rosenbrock = np.array([-1.0, 2.0])
        
        # Test functions to run
        self.test_functions = ['quadratic1', 'quadratic2', 'quadratic3', 
                              'rosenbrock', 'linear', 'exponential']
    
    def test_quadratic1(self):
        """Test optimization on quadratic function with circular contours."""
        self._run_optimization_test('quadratic1', self.x0_default, self.max_iter_default)
    
    def test_quadratic2(self):
        """Test optimization on quadratic function with elliptical contours.""" 
        self._run_optimization_test('quadratic2', self.x0_default, self.max_iter_default)
    
    def test_quadratic3(self):
        """Test optimization on quadratic function with rotated elliptical contours."""
        self._run_optimization_test('quadratic3', self.x0_default, self.max_iter_default)
    
    def test_rosenbrock(self):
        """Test optimization on Rosenbrock function."""
        self._run_optimization_test('rosenbrock', self.x0_rosenbrock, self.max_iter_default,
                                  max_iter_gd=self.max_iter_rosenbrock_gd)
    
    def test_linear(self):
        """Test optimization on linear function."""
        self._run_optimization_test('linear', self.x0_default, self.max_iter_default)
    
    def test_exponential(self):
        """Test optimization on exponential function."""
        self._run_optimization_test('exponential', self.x0_default, self.max_iter_default)
    
    def _run_optimization_test(self, func_name: str, x0: np.ndarray, max_iter: int,
                              max_iter_gd: int = None):
        """
        Run optimization test for a specific function.
        
        Args:
            func_name: Name of function to test
            x0: Initial point
            max_iter: Maximum iterations for Newton method
            max_iter_gd: Maximum iterations for Gradient Descent (if different)
        """
        print(f"\n{'='*60}")
        print(f"Testing {func_name.upper()} function")
        print(f"{'='*60}")
        
        func = FUNCTIONS[func_name]
        
        if max_iter_gd is None:
            max_iter_gd = max_iter
        
        # Test Gradient Descent
        print(f"\n--- GRADIENT DESCENT ---")
        minimizer_gd = UnconstrainedMinimizer()
        final_x_gd, final_f_gd, success_gd = minimizer_gd.minimize(
            func, x0, self.obj_tol, self.param_tol, max_iter_gd, 'gd'
        )
        path_x_gd, path_f_gd = minimizer_gd.get_path()
        
        print(f"Final result - GD: x = [{final_x_gd[0]:.6f}, {final_x_gd[1]:.6f}], "
              f"f(x) = {final_f_gd:.10f}, success = {success_gd}")
        
        # Test Newton's Method
        print(f"\n--- NEWTON'S METHOD ---")
        minimizer_newton = UnconstrainedMinimizer()
        final_x_newton, final_f_newton, success_newton = minimizer_newton.minimize(
            func, x0, self.obj_tol, self.param_tol, max_iter, 'newton'
        )
        path_x_newton, path_f_newton = minimizer_newton.get_path()
        
        print(f"Final result - Newton: x = [{final_x_newton[0]:.6f}, {final_x_newton[1]:.6f}], "
              f"f(x) = {final_f_newton:.10f}, success = {success_newton}")
        
        # Create plots
        self._create_plots(func_name, func, path_x_gd, path_f_gd, 
                          path_x_newton, path_f_newton)
        
        # Assertions for unittest
        self.assertIsInstance(final_x_gd, np.ndarray)
        self.assertIsInstance(final_f_gd, (int, float))
        self.assertIsInstance(success_gd, bool)
        self.assertIsInstance(final_x_newton, np.ndarray)
        self.assertIsInstance(final_f_newton, (int, float))
        self.assertIsInstance(success_newton, bool)
    
    def _create_plots(self, func_name: str, func, path_x_gd, path_f_gd,
                     path_x_newton, path_f_newton):
        """Create and save plots for the optimization results."""
        
        # Get plot limits and title
        x_limits, y_limits = get_plot_limits(func_name)
        title = get_function_title(func_name)
        
        # Plot 1: Contours with optimization paths
        paths = {
            'Gradient Descent': path_x_gd,
            'Newton Method': path_x_newton
        }
        
        fig1 = plot_contours_with_path(func, x_limits, y_limits, paths, 
                                      title + " - Optimization Paths")
        
        # Plot 2: Function values vs iterations
        function_values = {
            'Gradient Descent': path_f_gd,
            'Newton Method': path_f_newton
        }
        
        fig2 = plot_function_values(function_values, 
                                   f"{func_name.title()} - Function Value vs Iteration")
        
        # Save plots
        plt.figure(fig1.number)
        plt.savefig(f'{func_name}_contours_paths.png', dpi=300, bbox_inches='tight')
        print(f"Saved contour plot as {func_name}_contours_paths.png")
        
        plt.figure(fig2.number)
        plt.savefig(f'{func_name}_function_values.png', dpi=300, bbox_inches='tight')
        print(f"Saved function values plot as {func_name}_function_values.png")
        
        # Show plots
        plt.show()


def run_all_tests():
    """Run all optimization tests."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUnconstrainedMinimization)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("Running Numerical Optimization Tests")
    print("=" * 50)
    
    # Run all tests
    result = run_all_tests()
    
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
