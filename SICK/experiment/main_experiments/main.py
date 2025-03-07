import os
import importlib.util
import sys

current_file_path = os.path.abspath(__file__)

# Extract the file name from the full path
current_file_name = os.path.basename(current_file_path)
# Get the current script's directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# List all Python files in the directory except main.py
files_to_execute = [
    file for file in os.listdir(current_directory)
    if file.endswith(".py") and file != current_file_name

]
# Loop through all the Python files and execute them
for file in files_to_execute:
    file_path = os.path.join(current_directory, file)
    
    # Dynamically import and execute the Python file
    spec = importlib.util.spec_from_file_location(file, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[file] = module
    spec.loader.exec_module(module)
    
    print(f"Executed {file}")
