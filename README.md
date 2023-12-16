# Food Image Classifier 

### Introduction
This project is an in-depth study of computer vision algorithms used to classify five diverse food classes: apple pie, bruschetta, cupcakes, pizza, and sushi. Using a dataset from Kaggle with 1000 high-quality images per class, the project utilizes advanced preprocessing approaches such as normalization, standardization, and data augmentation. VGG16, an established deep learning model with exceptional results on image classification tasks, was chosen as the neural network architecture for the project. This project was implemented in python. 

### Objectives
The primary goal is to develop a robust model able to correctly identify images of the five chosen foods. This requires choosing an appropriate neural network architecture and optimizing it for the task. In addition, examining how normalization, standardization, and data augmentation affect the performance of the model along with understanding how these strategies affect the model's capacity to generalize different food images. Then examining the VGG16 model's capacity to generalize across food classes which includes evaluating its performance on a validation set and identifying potential areas for improvement.

### Experimental Methodology
The dataset acquired from Kaggle consists of 1000 high-quality images per class. The dataset represents a diverse food category. The data preprocessing pipeline includes normalization which scales the images to a range [0,1] to guarantee consistency of all images, then standardizing pixel values to 224x224 to accelerate model convergence during training and image transformations, such as, rotations, flips and zoom to augment the dataset which increases the training set’s diversity and lowering the likelihood of overfitting. 

### Evaluation Metrics
Key metrics, such as, accuracy, precision, recall and F1-score are used to evaluate model’s performance. 

## Collaborators 
[Ajay Kumar](https://github.com/ajaykumar4127) <br>
[Abel Mesfin](https://github.com/Abel0217)


