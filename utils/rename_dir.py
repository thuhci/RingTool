import os


def rename_json_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                new_name = file.replace('filtered', 'raw')
                
                old_file_path = os.path.join(root, file)
                new_file_path = os.path.join(root, new_name)
                
                os.rename(old_file_path, new_file_path)
                print(f'Renamed: {old_file_path} -> {new_file_path}')


directory = '<fill here>'

rename_json_files(directory)
