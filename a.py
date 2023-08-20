import os
import shutil

folder = './'

def remove_pycache(folder):
    for root, dirs, files in os.walk(folder):
        if '__pycache__' in dirs:
            pycache_folder = os.path.join(root, '__pycache__')
            print(f'Deleting {pycache_folder}')
            shutil.rmtree(pycache_folder)

remove_pycache(folder)