''' this is code to create a folder inside of images 
for each letter in the alphabet for efficiency purposes'''

import os

def create_alphabet_folders(base_directory):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    for letter in alphabet:
        folder_path = os.path.join(base_directory, letter)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Folder '{letter}' created at {folder_path}")

base_directory = '/Users/jonathanreyes/Desktop/asl-alphabet/images'
create_alphabet_folders(base_directory)

