import os
import tensorflow

input_dir = 'Food-samples-labeled'
output_dir = 'Food-samples-standardized2'

# Target size means what the output image resolution should be. In this case, it's 128x128.
target_size = (128, 128)

# Loop through each subdirectory in the main directory.
for subdir in os.listdir(input_dir):
    subdir_path = os.path.join(input_dir, subdir)
    
    if os.path.isdir(subdir_path):
        # Create new subdirectory in the output directory with the same name as the subdirectory from the input directory.
        output_subdir = os.path.join(output_dir, subdir)
        os.makedirs(output_subdir, exist_ok=True)

        # Loop through each image in a subdirectory.
        for filename in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, filename)
            
            # Load an image from the input subdirectory.
            img = tensorflow.keras.utils.load_img(file_path, target_size=target_size)
            img_array = tensorflow.keras.utils.img_to_array(img)
            
            # Save the resulting standardized image to the output subdirectory.
            output_path = os.path.join(output_subdir, filename)
            tensorflow.keras.utils.array_to_img(img_array).save(output_path)
