# config.py
import sys
import os

# Get the absolute path to the project directory
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
os.chdir(PROJECT_DIR)

# Add the utils directory to sys.path
UTILS_DIR = os.path.join(PROJECT_DIR, 'utils')
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)
