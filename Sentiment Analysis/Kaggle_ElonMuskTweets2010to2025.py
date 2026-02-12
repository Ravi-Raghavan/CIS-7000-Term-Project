import kagglehub
import shutil
import os
# Download latest version
path = kagglehub.dataset_download("dadalyndell/elon-musk-tweets-2010-to-2025-march")

print("Path to dataset files:", path)

destination_folder = os.getcwd()
print(f"Moving files to: {destination_folder}")

for item in os.listdir(path):
    source_item = os.path.join(path, item)
    destination_item = os.path.join(destination_folder, item)
    if os.path.isfile(source_item):
        shutil.move(source_item, destination_item)
        print(f"Moved: {source_item} to {destination_item}")
    else:
        print(f"Skipped (not a file): {source_item}")

