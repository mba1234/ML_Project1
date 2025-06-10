import sys
import os

print("--- DEBUGGING PYTHON PATH ---")
print(f"Current Working Directory: {os.getcwd()}")
print("sys.path (Python's module search path):")
for i, p in enumerate(sys.path):
    print(f"  {i}: {p}")

print("\n--- ATTEMPTING IMPORT ---")
try:
    from src.exception import CustomException
    print("SUCCESS: 'src.exception.CustomException' imported successfully from debug_path.py!")
except ModuleNotFoundError as e:
    print(f"FAILURE: ModuleNotFoundError: {e}")
    print("Hint: 'src' is likely not in sys.path or __init__.py files are missing.")
except Exception as e:
    print(f"ANOTHER ERROR: {type(e).__name__}: {e}")

print("--- DEBUGGING COMPLETE ---")