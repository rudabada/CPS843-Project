# Source code for copying directory is in https://www.geeksforgeeks.org/copy-a-directory-recursively-using-python-with-examples/
# Source code for renaming files is in https://pynative.com/python-rename-file/#h-rename-multiple-files-in-python
# Requirement: The Food-samples folder.
import os
import shutil  

def copyDirectory():
    # Source path  
    src = 'Food-samples/'
        
    # Destination path  
    dest = 'Food-samples-labeled/'
        
    # Copy the content of  
    # source to destination  
    destination = shutil.copytree(src, dest)

    # print(destination) prints the
    # path of newly created file 
    
def startRenamingProgram():
    main_directoryPath = r'Food-samples-labeled/'
    # count increase by 1 in each iteration
    # iterate all files from a directory
    for sub_directoryName in os.listdir(main_directoryPath):
        sub_directoryPath = main_directoryPath + sub_directoryName + "/"
        count = 1
        for old_fileName in os.listdir(sub_directoryPath):
            # Construct old file name
            source = sub_directoryPath + old_fileName

            # Adding the count to the new file name and extension
            destination = sub_directoryPath + sub_directoryName + " (" + str(count) + ").jpg"

            # Renaming the file
            os.rename(source, destination)
            count += 1
    print('All Files Renamed')
    
def renamingProgramMenu():
    while True:
        print("Type and enter \"start\" without the quotation mark to start program or enter \"stop\" without the quotation mark to stop program.")
        userInput = input("Enter your input: ")
        if userInput == "start":
            copyDirectory()
            startRenamingProgram()
        elif userInput == "stop":
            break
        else:
            print("Unrecognized or wrong input. Try again.")

renamingProgramMenu()
