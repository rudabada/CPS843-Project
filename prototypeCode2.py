# Time code source: https://www.freecodecamp.org/news/python-get-current-time/
# Time code source: https://www.programiz.com/python-programming/datetime/strftime

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import layers
from keras import models
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from datetime import datetime
import time


# Get epochs value from user input. Assume user inputs a valid positive integer.
userEpochsInput = input("Enter an epochs value: ")
epochs = int(userEpochsInput)

# Start program timer after user has entered their input.
start_time = time.time()

# Get current date and time.
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Program started at: " + current_time)

main_folder_path = 'Food-samples-labeled'

# Set image resolution
img_width = 224
img_height = 224

# Set batch size and number of classes
batch_size = 32
num_classes = 5

# Preprocessing, data augmentation and normalization. Also reads images from the different directories.
datagen = ImageDataGenerator(
    rescale=1./255,    # Standardize pixel values to [0,1]
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2    # Means 80% of the images will be trained on while 20% of the images will be validated or tested.
)

# Reads images from the different directories for model training.
train_generator = datagen.flow_from_directory(
    main_folder_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
# Reads images from the different directories for image validation.
validation_generator = datagen.flow_from_directory(
    main_folder_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Build VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

# Stops the model training.
base_model.trainable = False

# Configuration of a neural network's training process.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
#epochs = 10    # How many times a training dataset passes through the algorithm, in this case the VGG16 model.
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, epochs + 1)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel("Epoch Value")
plt.ylabel("Accuracy in Percentage Value")
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel("Epoch Value")
plt.ylabel("Loss")
plt.title('Training and Validation Loss')

#plt.show()
# Create a new subdirectory to store all the figures made by this program. If there is a subdirectory with the same name, overwrite its contents.
newSubdirectoryName = "Results of using " + str(epochs) + " epochs"
if os.path.isdir(newSubdirectoryName) == False:
    os.makedirs(newSubdirectoryName)
saveFileNameOfTrainingFigures = newSubdirectoryName + "/Figure_training.png"
plt.savefig(saveFileNameOfTrainingFigures)

# Evaluate the model on the test data
test_generator = datagen.flow_from_directory(
    main_folder_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

results = model.evaluate(test_generator)
print("Test Loss:", results[0])
print("Test Accuracy:", results[1])

# Class labels
class_labels = list(train_generator.class_indices.keys())

# Display all test images and predictions
all_test_images = []
all_test_labels = []

for _ in range(test_generator.samples // batch_size + 1):
    sample_images, sample_labels = next(test_generator)
    all_test_images.extend(sample_images)
    all_test_labels.extend(sample_labels)

# Convert to numpy arrays for compatibility with model.predict
all_test_images = np.array(all_test_images)
all_test_labels = np.array(all_test_labels)

# Predictions
all_test_predictions = model.predict(all_test_images)

# More evaluation metrics: precision, recall, f1-score, accuracy, support (amount of images), accuracy, macro average, and weighted average.
# Get class labels and predictions
class_labels = list(train_generator.class_indices.keys())
all_test_predictions_classes = np.argmax(all_test_predictions, axis=1)
true_classes = np.argmax(all_test_labels, axis=1)

# Generate classification report that includes the evaluation metrics.
report = classification_report(true_classes, all_test_predictions_classes, target_names=class_labels)

# Print and show the evaluation metrics.
print(report)
# Save the report to evaluation_metrics.txt file
evaluation_metricsFilePath = newSubdirectoryName + "\\evaluation_metrics.txt"
with open(evaluation_metricsFilePath, "w") as evaluation_metricsFile:
    evaluation_metricsFile.write(report)

# Display all test images and predictions, showing 10 images at a time
num_samples = len(all_test_images)
images_per_batch = 10
num_batches = int(np.ceil(num_samples / images_per_batch))

for batch_num in range(num_batches):
    start_idx = batch_num * images_per_batch
    end_idx = (batch_num + 1) * images_per_batch
    current_images = all_test_images[start_idx:end_idx]

    plt.figure(figsize=(15, 10))
    num_rows = 2
    num_cols = 5

    for i in range(len(current_images)):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(current_images[i])
        plt.axis('off')
        true_label = class_labels[np.argmax(all_test_labels[start_idx + i])]
        predicted_label = class_labels[np.argmax(all_test_predictions[start_idx + i])]
        plt.title(f'True: {true_label}\nPredicted: {predicted_label}')

    #plt.show()
    saveFileName = newSubdirectoryName + "/Figure" + str(batch_num + 1) + ".png"
    plt.savefig(saveFileName, dpi=300)



now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Results saved to: \"" + newSubdirectoryName + "\" directory.")
runtimeInMinutes = (time.time() - start_time) / 60
print("Program took " + str(runtimeInMinutes) + " minutes to run.")
