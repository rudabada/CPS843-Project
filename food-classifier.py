import os

directory = "food-100"
# definition for iterating over folders in a directory
def open_folders(directory):
    # returns a list of all entries (files, folders) in the directory
    for e in os.listdir(directory):
        ep = os.path.join(directory, e) # gets full path of each entry
        
        
            
#directory = "food-100"
#open_folders(directory)