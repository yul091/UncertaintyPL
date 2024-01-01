import os
import shutil
import random

def copy_random_java_files(src_root, dest_root, percentage=0.1):
    print("Copying files from {} to {}...".format(src_root, dest_root))
    for split in os.listdir(src_root):
        print("Processing split {} ...".format(split))
        split_path = os.path.join(src_root, split)
        if not os.path.isdir(split_path):
            continue
        for project in os.listdir(split_path):
            print("Processing project {} ...".format(project))
            project_path = os.path.join(split_path, project)
            if not os.path.isdir(project_path):
                continue
            all_java_files = [file for file in os.listdir(project_path) if file.endswith('.java')]
            num_files_to_select = max(1, int(percentage * len(all_java_files)))
            selected_files = random.sample(all_java_files, num_files_to_select)

            for file in selected_files:
                src_file_path = os.path.join(project_path, file)
                dest_file_path = os.path.join(dest_root, split, project, file)
                os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
                shutil.copy2(src_file_path, dest_file_path)

# Example usage:
shift = 'different_time'
src_root = f'data_original/main/{shift}'  # Source directory path
dest_root = f'data/main/{shift}'  # Destination directory path
copy_random_java_files(src_root, dest_root, percentage=0.1)
