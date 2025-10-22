import os                                                       # manages files and directories
import numpy as np                                              # numerical computing for matrices and arrays*
from PIL import Image                                           # image processing, PIL deprecated, belongs to Pillow library (pip install Pillow)
from sklearn.model_selection import train_test_split            # split dataset into training and testing sets
from sklearn.preprocessing import LabelBinarizer                # encode labels with value between 0 and n_classes-1
from sklearn.metrics import classification_report               # build a text report showing the main classification metrics
from sklearn.metrics import confusion_matrix                   # compute confusion matrix to evaluate the accuracy of a classification
import tensorflow as tf                                         
from tensorflow.keras.models import Sequential                  # linear stack of layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # layers for CNN
from tensorflow.keras.utils import to_categorical              # convert labels to categorical format
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

os.makedirs('NMDP', exist_ok=True)

# A numpy array is multidimensional data structure that allows to represent and manipulate data, like images, in Python. 
# It is extremely efficient for handling large amounts of numerical data.

# Load images from a directory and return them as a numpy array
def load_images(image_paths, label, image_size=(128, 128)):
    images = []
    labels = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(image_size)                    # Resize image to a fixed size
        image = np.array(image)                             # Convert image to numpy array
        images.append(image)                                # Append image to images list
        labels.append(label)                                # Append label to labels list
    return np.array(images), np.array(labels)

# Generator function that yields the path of each image in a directory
def impath(dir):
    for f in os.listdir(dir):
        if f.endswith('.jpg'):
            yield os.path.join(dir, f)

mediterranean_apex_dir = r"C:\Users\danym\OneDrive - Universita' degli Studi della Campania Luigi Vanvitelli\Università\Laurea Magistrale\First year material\Second semester\Machine Learning and AI\Amsterdam\Still_pictures_Mediterranean_ZIPfile_approx_40Mb\Apex Stills"
northeuropean_apex_dir = r"C:\Users\danym\OneDrive - Universita' degli Studi della Campania Luigi Vanvitelli\Università\Laurea Magistrale\First year material\Second semester\Machine Learning and AI\Amsterdam\Still_pictures_NorthEuropean_ZIPfile_approx_50Mb\Apex Stills"

mediterranean_image_paths = [f for f in impath(mediterranean_apex_dir)]
northeuropean_image_paths = [f for f in impath(northeuropean_apex_dir)]

mediterranean_images, mediterranean_labels = load_images(mediterranean_image_paths, label=0)
northeuropean_images, northeuropean_labels = load_images(northeuropean_image_paths, label=1)

images = np.concatenate([mediterranean_images, northeuropean_images], axis=0)
labels = np.concatenate([mediterranean_labels, northeuropean_labels], axis=0)

# Normalize pixel values to be between 0 and 1
images = images.astype("float32") / 255.0
print(type(images))
print(images)

# Binarization of labels: each label is converted to a binary vector with a 1 in the position of the label and 0 in the other positions
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Split the dataset into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(images, labels, test_size=0.25, random_state=42)

def create_compile(act1, act2):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation=act1),
        Dropout(0.5),
        Dense(2, activation=act2)
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])
    return model

# The model is a Neural Network with a sequential structure, that is a linear stack of layers (input, hidden, output) with a 
    # feedforward connection. A particular type of this kind of model is the Convolutional Neural Network (CNN) that is used for
    # image classification tasks. CNNs are particularly effective in recognizing patterns in images. The model is composed of:
    # - Conv2D(filters, kernel_size, activation, input_shape): convolutional layer that applies filters to the input image,
    # kernel_size is the size of the filter, activation is the activation function, input_shape is the shape of the input image 
    # (128x128 pixels and 3 channels (red, green, blue))
    # - MaxPooling2D(pool_size): pooling layer that reduces the spatial dimensions of the input image by taking the maximum value
    # in a pool of pixels of the specified size. It reduces the computational cost and the risk of overfitting
    # - Flatten(): flattening layer that converts the 2D matrix data to a vector
    # - Dense(units, activation): fully connected layer with units neurons and activation function
    # - Dropout(rate): at each iteration of the training, a percentage (rate) of neurons is randomly deactivated to prevent overfitting, 
    # losing some performance.


# Plotting function for metrics
def plot_metrics(history, metrics, act1, act2,i):
    plt.figure(figsize=(12, 12))
    
    plt.subplot(2, 1, 1)
    for metric in metrics:
        if metric in history.history:
            plt.plot(history.history[metric], label=metric)
    plt.xlabel('Epochs')
    plt.ylabel('Metric value')
    plt.title(f'Training Metrics\nActivation function 1: {act1}, Activation function 2: {act2}')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for metric in metrics:
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            plt.plot(history.history[val_metric], label=val_metric)
    plt.xlabel('Epochs')
    plt.ylabel('Metric value')
    plt.title(f'NMDP/Validation Metrics\nActivation function 1: {act1}, Activation function 2: {act2}')
    plt.legend()
    
    plt.tight_layout()

    plt.savefig(f'NMDP/Plot_metrics_epoch_{epoch}_act1_{act1}_act2_{act2}.png')
    plt.show()
    
    # Export metrics to CSV
    metrics_data = {metric: history.history[metric] for metric in metrics}
    val_metrics_data = {f'val_{metric}': history.history[f'val_{metric}'] for metric in metrics}
    metrics_data.update(val_metrics_data)
    df = pd.DataFrame(metrics_data)
    df.to_csv(f'NMDP/metrics_epoch_{epoch}_act1_{act1}_act2_{act2}.csv', index=False)

# Plotting function for confusion matrix
def plot_confusion_matrix(testY_true, predictions, act1, act2, epoch,i):
    cm = confusion_matrix(testY_true, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Mediterranean', 'North European'], yticklabels=['Mediterranean', 'North European'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix\nActivation function 1: {act1}, Activation function 2: {act2}, Epoch: {epoch}')
    plt.savefig(f'NMDP/Confusion_matrix_{epoch}_act1_{act1}_act2_{act2}.png')
    plt.show()

# Compiling and training the model with different activation functions and epochs
epochs = [30, 40]
i=0
for epoch in epochs:
    # Test also sigmoid, tanh, swish, softmax, gelu
    act1 = 'relu' 
    act2 = 'softmax'
    model = create_compile(act1=act1, act2=act2)
    history = model.fit(trainX, trainY, epochs=epoch, batch_size=32, validation_split=0.2)
    
    predictions = model.predict(testX)
    predictions = np.argmax(predictions, axis=1)
    testY_true = np.argmax(testY, axis=1)
    
    report = classification_report(testY_true, predictions, target_names=['Mediterranean', 'North European'])
    if epoch == 10:
        print(report)
    
    metrics = ['accuracy', 'Precision', 'Recall', 'loss'] 
    i=i+1
    plot_metrics(history, metrics, act1, act2,i)
    plot_confusion_matrix(testY_true, predictions, act1, act2, epoch,i)

model.save('trained_model.keras')

