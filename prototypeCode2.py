import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import layers
from keras import models
import matplotlib.pyplot as plt

main_folder_path = 'Food-samples-labeled'

# Set image resolution
img_width = 224
img_height = 224

# Set batch size and number of classes
batch_size = 32
num_classes = 5

# Data augmentation and normalization. Also reads images from the different directories.
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
epochs = 10
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
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

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

# Display all test images and predictions, showing 10 images at a time
num_samples = len(all_test_images)
images_per_batch = 10
num_batches = int(np.ceil(num_samples / images_per_batch))

for batch_num in range(num_batches):
    start_idx = batch_num * images_per_batch
    end_idx = (batch_num + 1) * images_per_batch
    current_images = all_test_images[start_idx:end_idx]

    plt.figure(figsize=(15, 5))
    num_rows = 2
    num_cols = 5

    for i in range(len(current_images)):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(current_images[i])
        plt.axis('off')
        true_label = class_labels[np.argmax(all_test_labels[start_idx + i])]
        predicted_label = class_labels[np.argmax(all_test_predictions[start_idx + i])]
        plt.title(f'True: {true_label}\nPredicted: {predicted_label}')

    plt.show()
