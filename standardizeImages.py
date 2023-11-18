import os
import tensorflow

input_dir = 'Food-samples-labeled'
output_dir = 'Food-samples-standardized2'

# Target size resolution
target_size = (128, 128)

for subdir in os.listdir(input_dir):
    subdir_path = os.path.join(input_dir, subdir)
    
    if os.path.isdir(subdir_path):
        output_subdir = os.path.join(output_dir, subdir)
        os.makedirs(output_subdir, exist_ok=True)

        for filename in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, filename)
            
            # Load image
            img = tensorflow.keras.utils.load_img(file_path, target_size=target_size)
            img_array = tensorflow.keras.utils.img_to_array(img)
            
            # Save the standardized image to the output subdirectory
            output_path = os.path.join(output_subdir, filename)
            tensorflow.keras.utils.array_to_img(img_array).save(output_path)