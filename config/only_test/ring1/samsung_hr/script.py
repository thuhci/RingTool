import os

def rename_json_files(directory):
    # Walk through all files in the given directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is a .json file
            if file.endswith('.json'):
                # Check if "oura_hr" is in the file name
                if 'oura_hr' in file:
                    # Create the new file name by replacing 'oura_hr' with 'samsung_hr'
                    new_name = file.replace('oura_hr', 'samsung_hr')
                    
                    # Get the full file path
                    old_file_path = os.path.join(root, file)
                    new_file_path = os.path.join(root, new_name)
                    
                    # Rename the file
                    os.rename(old_file_path, new_file_path)
                    print(f'Renamed: {old_file_path} -> {new_file_path}')

# Specify the directory to search
directory = '/home/dykderrick/mega/code/RingTool/config/only_test/ring1/samsung_hr'

# Run the function to rename files
rename_json_files(directory)
