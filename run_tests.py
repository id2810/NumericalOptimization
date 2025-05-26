#!/usr/bin/env python3
"""
Simple test runner for the numerical optimization homework.
Runs all tests and generates the required plots.
"""

import sys
import os
import numpy as np

# Add the parent directory to Python path so we can import from src and tests
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import and run tests
from tests.test_unconstrained_min import run_all_tests

if __name__ == "__main__":
    print("Numerical Optimization - Programming Assignment 01")
    print("=" * 55)
    print()
    
    # Run all tests
    result = run_all_tests()
    
    print("\n" + "=" * 55)
    print("Test execution completed!")
    print(f"Check the generated PNG files for plots.")
    print("=" * 55)
