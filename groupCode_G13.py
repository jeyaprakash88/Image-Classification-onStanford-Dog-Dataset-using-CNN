#!/usr/bin/env python
# coding: utf-8

# <h3 style="text-align: center; font-family: 'Garamond'; font-size:50px">🐾 Fine-Grained Image Classification On Stanford Dog Dataset Using CNN 🐾</h3>

# [1. EXPLORATORY DATA ANALYSIS](#1)  
#   
# [2. DATA VISUALIZATION](#5)
# 
# [3. STATISTICAL ANALYSIS](#3)
# 
# [4. SPLITTING OF DATASET](#2)
# 
# [5. DATA PRE-PROCESSING ](#4)
# 
# [6. DATA AUGMENTATION](#6)
# 
# [7. MODEL DEVELOPMENT & IMPLEMENTATION](#7)<blockquote>
#    [7.1. Sequential Model](#71)<blockquote>
#        [7.1.1 Base Model](#711)   
#        [7.1.2 Model with More Layers & Dropout](#712)   
#        [7.1.3 Model with Some hyperparameter tuning](#713)<blockquote>
#             [7.1.3.1 Model Testing & Result Analysis](#7131)   
#             [7.1.3.2 Error Analysis](#7132)</blockquote></blockquote></blockquote><blockquote>
#    [7.2. VGG16 Model](#72)<blockquote>
#        [7.2.1 Base Model](#721)   
#        [7.2.2 Model with More Layers & Dropout](#722)<blockquote>
#             [7.2.2.1 Model Testing & Result Analysis](#7221)   
#             [7.2.2.2 Error Analysis](#7222)</blockquote></blockquote></blockquote><blockquote>
#    [7.3. Xception Model](#73)<blockquote>
#        [7.3.1 Model with hyperparameters](#731)<blockquote>
#             [7.3.1.1 Model Testing & Result Analysis](#7311)   
#             [7.3.1.2 Error Analysis](#7312)</blockquote></blockquote></blockquote><blockquote>
#    [7.4. InceptionResNetV2 Model](#74)<blockquote>
#        [7.4.1 Model with hyperparameters](#741)<blockquote>
#             [7.4.1.1 Model Testing & Result Analysis](#7411)   
#             [7.4.1.2 Error Analysis](#7412)</blockquote></blockquote></blockquote><blockquote>
#    [7.5. InceptionV3 Model](#75)<blockquote>
#        [7.5.1 Model with hyperparameters](#751)<blockquote>
#             [7.5.1.1 Model Testing & Result Analysis](#7511)   
#             [7.5.1.2 Error Analysis](#7512)</blockquote></blockquote></blockquote><blockquote>
#    [7.6. Ensemble of Pretrained Models](#76)<blockquote>
#        [7.6.1 Loading the Whole dataset](#761)   
#        [7.6.2 Analysis of the Dataset](#762)   
#        [7.6.3 Create a dataframe for train and test seperately](#763)   
#        [7.6.4 Data Pre-processing](#763)   
#        [7.6.5 Feature Extraction](#765)   
#        [7.6.6 Model Creation](#766)<blockquote>
#             [7.6.6.1 Test Feature Extraction](#7661)   
#             [7.6.6.2 Model Testing & Result Analysis](#7662)   
#             [7.6.6.3 Error Analysis](#7663)</blockquote></blockquote></blockquote>
# 
# [8. TESTING WITH CUSTOM INPUT](#8)
# 
# _______________________________
# 

# In[1]:


#Importing required libraries

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import shutil
import gc
import random
import seaborn as sns
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import FitFailedWarning 

from tqdm.autonotebook import tqdm

from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras.utils import load_img
from keras import models, layers
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model


from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Lambda, Input, BatchNormalization
from keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


# ## Load the data :

# In[2]:


# Give the file path for the image folders where it is stored

file_path = "C:/Users/c22081255/OneDrive - Cardiff University/Desktop/Data Science and Analytics/CMT307 Applied Machine Learning/Course_work_2/Images"


# ## Breed Selection by semantic features

# In[3]:


import nltk
from nltk.corpus import wordnet as wn

# Set path to the directory containing the raw images of the Stanford Dog dataset
dataset_path = file_path

# Load a list of all the dog breeds in the dataset
breeds = os.listdir(dataset_path)

# Select only the specified number of dog breeds
num_breeds = 120
selected_breeds = np.random.choice(breeds, num_breeds, replace=False)

# Define a function to extract semantic features for a breed name
def extract_semantic_features(breed):
    synsets = wn.synsets(breed, pos=wn.NOUN)
    synonyms = set()
    hypernyms = set()
    hyponyms = set()
    for synset in synsets:
        synonyms.update(synset.lemma_names())
        hypernyms.update([x.name().split(".")[0] for x in synset.hypernyms()])
        hyponyms.update([x.name().split(".")[0] for x in synset.hyponyms()])
    return synonyms, hypernyms, hyponyms

# Compute pairwise semantic similarity between all breeds
breed_similarities = np.zeros((num_breeds, num_breeds))
for i in range(num_breeds):
    for j in range(num_breeds):
        if i == j:
            breed_similarities[i, j] = 1.0
        else:
            syn_i, hyp_i, hypo_i = extract_semantic_features(selected_breeds[i])
            syn_j, hyp_j, hypo_j = extract_semantic_features(selected_breeds[j])
            similarity = len(syn_i.intersection(syn_j)) + len(hyp_i.intersection(hyp_j)) + len(hypo_i.intersection(hypo_j))
            breed_similarities[i, j] = similarity

# Select 10 most dissimilar breeds
selected_indices = np.argsort(-np.sum(breed_similarities, axis=1))[:10]
selected_breeds = [selected_breeds[i] for i in selected_indices]

print("Selected breeds:")
print(selected_breeds)


# In[4]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Set number of images to display for each breed
num_images = 2

# Define a function to plot images for a given breed
def plot_breed_images(breed):
    breed_path = os.path.join(dataset_path, breed)
    breed_images = os.listdir(breed_path)[:num_images]
    fig, ax = plt.subplots(1, num_images, figsize=(12, 4))
    for i, image_name in enumerate(breed_images):
        image_path = os.path.join(breed_path, image_name)
        image = mpimg.imread(image_path)
        ax[i].imshow(image)
        ax[i].axis("off")
    fig.suptitle(breed, fontsize=16)
    plt.show()

# Plot images for each of the selected breeds
for breed in selected_breeds:
    plot_breed_images(breed)


# ### Initially working with only 10 dog breeds

# In[5]:


# reading the path name, Breed name, label of the each image 
breeds = [breed.split('-',1)[1] for breed in selected_breeds]

from itertools import chain
m, y, z, label = [], [], [], []
fullpaths = [file_path+'/{}'.format(dog_class) for dog_class in selected_breeds]
for counter, fullpath in enumerate(fullpaths):
    for imgname in os.listdir(fullpath):
        m.append([fullpath+'/'+imgname])
        y.append(breeds[counter])
        label.append(imgname.split('.')[0])
m=list(chain.from_iterable(m))


# In[6]:


# to get the image size

for i in range(len(m)):
    with Image.open(m[i]) as img:
        z.append(np.array(img).shape)


# In[7]:


# creating a dataframe with 'Path', 'id', 'Breed_Name', 'Image_size'

combined = list(zip(m,label,y, z))
m[:], label[:], y[:], z[:] = zip(*combined)
df = pd.DataFrame(combined, columns =['Path', 'id', 'breed', 'Image_size'])
df['Height'] = df['Image_size'].apply(lambda m: m[0])
df['Width'] = df['Image_size'].apply(lambda m: m[1])


# In[8]:


df.head()


# #### Creating label as train and test

# In[9]:


# Getting the Dog breed name
Breeds = df['breed'].unique()

# Get indices of rows for each breed name
for category in Breeds:
    breed_indices = df.index[df['breed'] == category]

    # Split indices into train, test sets with 70%, 30% respectively
    train_indices, test_indices = train_test_split(breed_indices, test_size=0.3, random_state=42)

    # Create a new column in the dataframe with the label
    df.loc[train_indices, 'label'] = 'train'
    df.loc[test_indices, 'label'] = 'test'
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
df['Target'] = labelencoder.fit_transform(df['breed'])
df


# ## 1.Exploratory Data Analysis by analyzing the dataset<a name="1"></a>

# In[10]:


# Checking the shape of the dataset:
df.shape


# In[11]:


# Checking the data type of each columns and non-null count:
df.info()


# In[12]:


#Find the duplicates
df.duplicated().sum()


# In[13]:


df.describe()


# In[14]:


df.describe(include='object')


# In[15]:


# to find the counts of the each breed

df['breed'].value_counts()


# ## 2. EDA By visualization<a name="5"></a>

# In[16]:


#plotting the breeds againt the count

plt.figure(figsize=(10,5))
chart = sns.countplot(data=df, x='breed', palette='Set1')
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()


# In[17]:


# create scatter plot
plt.scatter(df['Width'], df['Height'])

# add axis labels and title
plt.xlabel('Image Width')
plt.ylabel('Image Height')
plt.title('Scatter Plot of Image Dimensions')

# show the plot
plt.show()


# In[18]:


# create a bar chart of image height and image width
heights = df['Height']
widths = df['Width']
labels = df['Path']

x = list(range(len(df)))  # convert range to list
width = 0.35

fig, ax = plt.subplots(figsize=(30, 15))
rects1 = ax.bar([i - width/2 for i in x], heights, width, label='Image Height')
rects2 = ax.bar([i + width/2 for i in x], widths, width, label='Image Width')

ax.set_ylabel('Pixels')
ax.set_title('Image Size by Path')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()


# ## 3.STATISTICAL ANALYSIS<a name="3"></a>

# In[19]:


# Encode categorical variables, if any
new_df = pd.get_dummies(df, columns=['breed'])

# Compute summary statistics
summary_stats = new_df.describe()

# Print the summary statistics
print(summary_stats)


# In[20]:


# Displaying images randomly

from skimage.io import imread

plt.figure(figsize=(18,18))
for counter, i in enumerate(random.sample(range(0, len(m)), 28)):
    plt.subplot(7,5, counter + 1)
    plt.subplots_adjust(hspace=0.3)
    filename = m[i]
    image = imread(filename)
    plt.imshow(image)
    plt.title(y[i], fontsize = 12)
plt.show()


# #### Mean / SD Pixel Intensity

# In[26]:


# Assume 'img_paths' is a list of paths to your image dataset
pixel_intensities = []

# Loop through all the images in the dataset
for img_path in m:
    # Read the image and convert it to grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Flatten the 2D array into a 1D array of pixel intensities
    pixel_intensities.extend(img.ravel())

# Calculate the mean and standard deviation of the pixel intensities
mean = np.mean(pixel_intensities)
std_dev = np.std(pixel_intensities)

print("Mean pixel intensity:", mean)
print("Standard deviation of pixel intensity:", std_dev)


# #### SIFT (Scale-Invariant Feature Transform)

# In[22]:


import cv2
import random

''' SIFT (Scale-Invariant Feature Transform) is a computer vision algorithm for detecting and describing local features 
    in images. It was introduced by David Lowe in 1999 and has become a widely used algorithm for various computer vision tasks,
    such as object recognition, image stitching, and 3D reconstruction'''

# Assuming x is a list of valid image filenames

for counter, i in enumerate(random.sample(range(0, len(m)), 1)):
    filename = m[i]
    image = cv2.imread(filename)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the SIFT feature detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Draw the keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

    # Display the image with keypoints
    cv2.imshow('Image with Keypoints', image_with_keypoints)

    # Wait for a key press event to occur
    cv2.waitKey(0)

# Destroy all open windows
cv2.destroyAllWindows()


# In[27]:


import numpy as np
import cv2
import os

# Define the directory containing the images
directory = file_path

# Define the target size for resizing the images
target_size = (150, 150)

# Initialize variables for statistics
num_images = 0
mean_image = None
stddev_image = None

# Loop over all images in the directory
for i in range(len(df)):
    filename = df['Path'][i]
    if filename.endswith('.jpg'):
        # Load the image
        image = cv2.imread(os.path.join(directory, filename))
        
        # Resize the image to the target size
        image = cv2.resize(image, target_size)
        
        # Convert the image to float
        image = image.astype(np.float32)

        # Compute the mean and standard deviation
        if mean_image is None:
            mean_image = np.zeros_like(image)
        if stddev_image is None:
            stddev_image = np.zeros_like(image)
        num_images += 1
        delta = image - mean_image
        mean_image += delta / num_images
        stddev_image += delta * (image - mean_image)**2
# Compute the final statistics
stddev_image = np.sqrt(stddev_image / (num_images - 1))
mean_image /= num_images

# Print the statistics
print('Number of images:', num_images)
print('Mean image:', mean_image)
print('Standard deviation image:', stddev_image)


# The output shows that there are 1919 images in the dataset, and the mean image has values close to zero for each channel. This indicates that the pixel values of the images are centered around zero.
# 
# The standard deviation image has small values, which suggests that the pixel values of the images are not highly variable. This could be due to the nature of the images themselves or the fact that they have been preprocessed (e.g., normalized) before being used in this analysis.
# 
# Overall, the mean image and standard deviation image can be useful for gaining insights into the characteristics of the image dataset and for normalizing the pixel values of the images before training a machine learning model.

# In[28]:


#Converting Grayscale

from skimage.color import rgb2gray
from matplotlib.image import imread
from skimage import filters
for counter, i in enumerate(random.sample(range(0, len(m)), 1)):
    filename = m[i]
    image = imread(filename)
    grayscale = rgb2gray(image)#Edge Detection
    ed_sobel = filters.sobel(grayscale)
    plt.imshow(ed_sobel, cmap='Greys');


# In[29]:


#Plotting the Image and the Histogram of gray values

from skimage.exposure import histogram
hist, hist_centers = histogram(grayscale)
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(grayscale, cmap=plt.cm.gray)
axes[0].set_axis_off()
axes[1].plot(hist_centers, hist, lw=2)
axes[1].set_title('histogram of gray values')


# In[18]:


labels = df[['id','breed']][(df["label"] == 'train')].reset_index(drop=True)
labels.head()


# In[19]:


labels.describe()


# In[20]:


#function to show bar length
def bar_chart(ax): 
    
    for p in ax.patches:
        val = p.get_width() #height of the bar
        x = p.get_x()+ p.get_width() # x- position 
        y = p.get_y() + p.get_height()/2 #y-position
        ax.annotate(round(val,2),(x,y))
        
#finding top dog brands

plt.figure(figsize = (10,3))
ax0 =sns.countplot(y=labels['breed'],order=labels['breed'].value_counts().index)
bar_chart(ax0)
plt.show()


# ## 4.SPLITTING OF DATASET<a name="2"></a>

# In[21]:


train = df[['Path', 'id','breed']][(df["label"] == 'train')].reset_index(drop=True)
test = df[['Path', 'id','breed']][(df["label"] == 'test')].reset_index(drop=True)

train_x = train.iloc[:, 0]
train_y = train.iloc[:, 1:]


test_x = test.iloc[:, 0]
test_y = test.iloc[:, 1:]


# In[22]:


#Create list of alphabetically sorted labels.

classes = sorted(list(set(labels['breed'])))
n_classes = len(classes)
print('Total unique breed {}'.format(n_classes))

#Map each label string to an integer label.
breed_to_num = dict(zip(classes, range(n_classes)))
breed_to_num


# ### 5.DATA PRE-PROCESSING<a name="4"></a>

# In[11]:


input_shape = (150,150,3)

# convert the image to array for train image and the label

def images_to_array(directory, label_dataframe, target_size = input_shape):
    
    image_labels = label_dataframe['breed']
    images = np.zeros([len(label_dataframe), target_size[0], target_size[1], target_size[2]],dtype=np.uint8) #as we have huge data and limited ram memory. uint8 takes less memory
    y = np.zeros([len(label_dataframe),1],dtype = np.uint8)
    
    for ix in range(len(directory)):
        img_dir = os.path.join(directory[ix])
        img = load_img(img_dir, target_size = target_size)
        images[ix]=img
        del img
        
        dog_breed = image_labels[ix]
        y[ix] = breed_to_num[dog_breed]
    print('Input Data Size: ', images.shape)
    
    y = to_categorical(y)
    
    return images,y


# In[12]:


#Function to read images from test directory
img_size = (150,150,3)

def images_to_array_test(test_path, img_size = (150,150,3)):
    
    data_size = len(test_path)
    images = np.zeros([data_size, img_size[0], img_size[1], 3], dtype=np.uint8)
    
    
    for ix in range(data_size):
        img_dir = os.path.join(test_path[ix])
        img = load_img(img_dir, target_size = img_size)
        images[ix]=img
        del img
    print('Ouptut Data Size: ', images.shape)
    return images


# In[25]:


import time 
t = time.time()

X,y = images_to_array(train_x, train_y)

print('runtime in seconds: {}'.format(time.time() - t))


# In[26]:


test_data = images_to_array_test(test_x, img_size)
X_test,y_test = images_to_array(test_x, test_y)


# In[27]:


# lets check some dogs and their breeds
n=25

# setup the figure 
plt.figure(figsize=(20,20))

for i in range(n):
#     print(i)
    ax = plt.subplot(5, 5, i+1)
    plt.title(classes[np.where(y[i] ==1)[0][0]])
    plt.imshow(X[i].astype('int32')) # .astype('int32') ---> as imshow() needs integer data to read the image


# ## 6.DATA AUGMENTATION<a name="6"></a>

# In[28]:


# assume x_train and y_train are the training data
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the image size and batch size for the data generator
img_size = (150, 150)
batch_size = 32

# Define the data generator for the train, validation, and test sets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Define the data generator for the train and test sets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# define the validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)

# create validation generator
val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size)

# compute the validation steps
validation_steps = val_generator.n // val_generator.batch_size

# Create the data generators for the train and test sets
batch_size = 32
train_generator = train_datagen.flow(
    X,
    y,
    batch_size=batch_size,
    shuffle=True,
    seed=42)

test_generator = test_datagen.flow(
    X_test,
    y_test,
    batch_size=batch_size,
    shuffle=False)


# In[29]:


img_id = 2
dog_generator = train_datagen.flow(x_train[img_id:img_id+1], y_train[img_id:img_id+1],
                                  shuffle = False, batch_size = batch_size, seed = 1)

plt.figure(figsize=(20,20))
dogs = [next(dog_generator) for i in range(0,5)]
for counter, dog in enumerate(dogs):
    plt.subplot(1,5, counter+1)
    plt.imshow(dog[0][0])
plt.show()


# ## 7.MODEL DEVELOPMENT & IMPLEMENTATION<a name="7"></a>

# ### 7.1. Sequential Model<a name="71"></a>

# <p><b>Convolutional Neural Networks (CNNs)</b></br></br>
# Convolutional Neural Networks (CNNs) are a types of deep neural networks that are commonly used for image recognition tasks. They work by learning to recognize patterns in images through a series of convolutional layers, pooling layers, and fully connected layers.
# 
# How do CNNs work?</br></br>
# <li>Convolutional layers: The first layer of a CNN applies filters to the input image to extract features such as edges, corners, and other visual patterns.</li>
# <li>Activation function: After applying the filters, an activation function is used to introduce non-linearity into the output of the convolutional layer.</li>
# <li>Pooling layers: The output of the convolutional layer is then passed through a pooling layer that reduces the dimensionality of the output while retaining important information. This helps to make the model more robust to small changes in the input image.</li>
# <li>Fully connected layers: The final layer of the CNN is a fully connected layer that takes the output of the convolutional and pooling layers and produces the final classification output.</li></p>

# ### 7.1.1 Base Model<a name="711"></a>

# In[30]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Define the input shape of the images
input_shape = (150, 150, 3)

# Create a sequential model
model = Sequential()

# Add the first convolutional layer with 32 filters, a kernel size of 3x3, and the ReLU activation function
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

# Add the first max pooling layer with a pool size of 2x2
model.add(MaxPooling2D((2, 2)))

# Add the second convolutional layer with 64 filters, a kernel size of 3x3, and the ReLU activation function
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add the second max pooling layer with a pool size of 2x2
model.add(MaxPooling2D((2, 2)))

# Add the third convolutional layer with 128 filters, a kernel size of 3x3, and the ReLU activation function
model.add(Conv2D(128, (3, 3), activation='relu'))

# Add the third max pooling layer with a pool size of 2x2
model.add(MaxPooling2D((2, 2)))

# Add the fourth convolutional layer with 256 filters, a kernel size of 3x3, and the ReLU activation function
model.add(Conv2D(256, (3, 3), activation='relu'))

# Add the fourth max pooling layer with a pool size of 2x2
model.add(MaxPooling2D((2, 2)))

# Flatten the output of the previous layer
model.add(Flatten())

# Add a dense layer with 512 neurons and the ReLU activation function
model.add(Dense(512, activation='relu'))

# Add a dropout layer with a rate of 0.5
model.add(Dropout(0.5))

# Add a dense layer with the number of neurons equal to the number of dog breeds to classify and the softmax activation function
model.add(Dense(n_classes, activation='softmax'))


# In[31]:


# Compile the model with categorical cross-entropy loss, the Adam optimizer, and accuracy as the metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[32]:


print(model.summary())


# In[20]:


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# In[33]:


# Train the model with regularization and early stopping callbacks

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.n // batch_size,
                              epochs=30,
                              validation_data=val_generator,
                              validation_steps=validation_steps,
                              callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6),
                                         EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])


# In[34]:


# Evaluate the model on the test set

test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.n//test_generator.batch_size)

# print the test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
# Evaluate the model on the test set
scores = model.evaluate(X_test, y_test, verbose=0)

# Print the mean accuracy
print("Mean accuracy: %.2f%%" % (scores[1]*100))


# In[35]:


# Plot the training and validation accuracy values

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot the training and validation loss values

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# <li>Increase the model capacity: You can increase the number of layers or the number of units in each layer of your model to improve its capacity to learn from the data. This may help your model to capture more complex patterns in the data.</li>
# <br>
# <li>Augment the training data: You can apply various data augmentation techniques, such as rotation, scaling, or flipping, to the training data to create more examples and help the model learn to generalize better.</li>
# <br>
# <li>Adjust the learning rate: The learning rate determines how quickly the model updates its weights during training. If the learning rate is too high, the model may overshoot the optimal weights and fail to converge. If it is too low, the model may take too long to converge. You can experiment with different learning rates to find the optimal value for your model.</li>
# <br>
# <li>Try a different optimizer: The optimizer determines the strategy that the model uses to update its weights during training. Different optimizers may work better for different types of problems. You can try different optimizers, such as Adam or SGD, to see if they improve the performance of your model.</li>
# <br>
# <li>Adjust the batch size: The batch size determines how many examples the model processes at once during training. A smaller batch size may help the model converge faster and generalize better, but it can also slow down training. You can experiment with different batch sizes to find the optimal value for your model.</li>

# ### 7.1.2 Model with More Layers & Dropout<a name="712"></a>

# In[36]:



# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# In[37]:


# Compile the model with categorical crossentropy and RMSprop optimizer

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# In[23]:


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# In[ ]:


# Train the model with fit_generator

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=30,
    validation_data=val_generator,
    validation_steps=validation_steps)


# In[ ]:


# Evaluate the model on the test set

test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.n//test_generator.batch_size)

# print the test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


# In[ ]:


# Plot the training and validation accuracy values
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot the training and validation loss values
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# #### 7.1.3 Model with Some hyperparameter tuning<a name="713"></a>

# In[38]:


# Build the model architecture

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# In[39]:


# Compile the model with categorical crossentropy and RMSprop optimizer

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# In[40]:


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# In[41]:


# Train the model with fit_generator

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=30,
    validation_data=val_generator,
    validation_steps=validation_steps)


# In[42]:


# Plot the training and validation accuracy values
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot the training and validation loss values
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# #### 7.1.3.1 Model Testing & Result analysis<a name="7131"></a>

# In[43]:


from sklearn.metrics import confusion_matrix
# Evaluate the model on the test set

test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.n//test_generator.batch_size)

# print the test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
# Evaluate the model on the test set

# make predictions on test data
y_pred_proba = model.predict(test_generator)
# convert probabilities to class labels
y_pred = np.argmax(y_pred_proba, axis=1)

# get true labels
y_true = np.argmax(y_test, axis=1)


print(classification_report(y_true,y_pred ))

# create confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# plot the confusion matrix
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Dog Breed Classification - Model 1')
plt.colorbar()
tick_marks = np.arange(len(breed_to_num))
plt.xticks(tick_marks, breed_to_num.keys(), rotation=90)
plt.yticks(tick_marks, breed_to_num.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# add the counts to the plot
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.show()


# In[44]:


# switch the keys and values of the original dictionary
num_to_breed = {v: k for k, v in breed_to_num.items()}

# convert numerical labels to breed names for target labels and predictions
target_breeds = [num_to_breed[num] for num in y_true]
prediction_breeds = [num_to_breed[num] for num in y_pred]

predict_df = pd.DataFrame({'Target_Labels':target_breeds, 'predictions': prediction_breeds})
predict_df


# In[45]:


plt.figure(figsize = (30,50))
for counter, i in enumerate(random.sample(range(0, len(y_test)), 100)):
    plt.subplot(20, 5, counter+1)
    plt.subplots_adjust(hspace=0.6)
    actual = str(target_breeds[i])
    predicted = str(prediction_breeds[i])
    confidence = str(max(y_pred_proba[i]))
    plt.imshow(X_test[i]/255.0)
    plt.axis('off')
    plt.title('Actual :'+actual+'\nPredict :'+predicted+'\nConfidence :'+confidence, fontsize=12)
plt.show()


# #### 7.1.3.2 Error Analysis<a name="7132"></a>

# In[65]:


predict_df['count'] = 1
misclass_df = predict_df[predict_df['Target_Labels'] != predict_df['predictions']].groupby(['Target_Labels', 'predictions']).sum().sort_values(['count'], ascending=False).reset_index()

#Create a column to display breed pairs
misclass_df['pair'] = misclass_df['Target_Labels'] + ' / ' + misclass_df['predictions']

#Select the top 30 misclassified breed pairs
misclass_df = misclass_df[['pair', 'count']].head(30)

#Sort the dataframe by count
misclass_df = misclass_df.sort_values(['count'])

#Create horizontal bar plot
fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(misclass_df['pair'], misclass_df['count'], align='center')
ax.invert_yaxis()
ax.set_xlabel('Misclassified Count')
ax.set_ylabel('Breed Pair')
ax.set_title('Top 30 Misclassified Breed Pairs')
plt.show()


# In[66]:


# Analyze misclassified images
misclassified_indices = np.where(y_pred != y_true)[0]
misclassified_images = test_data[misclassified_indices]
misclassified_true_labels = y_true[misclassified_indices]
misclassified_pred_labels = y_pred[misclassified_indices]

for i in range(len(misclassified_indices)):
    image = misclassified_images[i]
    true_label = misclassified_true_labels[i]
    pred_label = misclassified_pred_labels[i]
    plt.imshow(image)
    plt.title(f'True label: {true_label}, Predicted label: {pred_label}')
    plt.show()


# <p>The first trial in building this model is building a CNN from scratch, without the use of transfer learning. Based on what we know about the purpose of transfer learning, it is likely that our results will be poor. </p>

# ### 7.2 VGG 16 Model<a name="72"></a>

# #### 7.2.1 Base Model<a name="721"></a>

# In[67]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16

# Define input shape
input_shape = (150, 150, 3)

# Load pre-trained VGG16 model with imagenet weights
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the pre-trained layers
for layer in vgg16.layers:
    layer.trainable = False

# Add a custom classification head
custom_head_output = Flatten()(vgg16.output)
custom_head_output = Dense(256, activation='relu')(custom_head_output)
custom_head_output = Dense(10, activation='softmax')(custom_head_output)

# Define the model
model = Model(inputs=vgg16.input, outputs=custom_head_output)

# Compile the model with categorical cross-entropy loss and Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping callback
early_stop = EarlyStopping(patience=5, monitor='val_loss')

# Print the model summary
model.summary()


# In[28]:


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# In[ ]:


# Train the model with early stopping

train_steps_per_epoch = x_train.shape[0]//batch_size
val_steps_per_epoch = x_val.shape[0]//batch_size
history = model.fit_generator(train_generator,
                              steps_per_epoch=train_steps_per_epoch,
                              validation_data=val_generator,
                              validation_steps=val_steps_per_epoch,
                              epochs = 30, verbose=1,
                              callbacks=[early_stop])


# In[ ]:


# Evaluate the model on the test set

test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.n//test_generator.batch_size)

# print the test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
# Evaluate the model on the test set


# In[ ]:


# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# #### 7.2.2 Model with More Layers & Dropout <a name="722"></a>

# In[30]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Define input shape
input_shape = (150, 150, 3)

# Load pre-trained VGG16 model with imagenet weights
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the pre-trained layers
for layer in vgg16.layers:
    layer.trainable = False

# Add a custom classification head
custom_head_output = Flatten()(vgg16.output)
custom_head_output = Dense(256, activation='relu')(custom_head_output)
custom_head_output = Dropout(0.5)(custom_head_output)  # Add Dropout layer with dropout rate of 0.5
custom_head_output = Dense(10, activation='softmax')(custom_head_output)

# Define the model
model = Model(inputs=vgg16.input, outputs=custom_head_output)

# Compile the model with categorical cross-entropy loss and Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping callback
early_stop = EarlyStopping(patience=5, monitor='val_loss')


# In[30]:


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# In[31]:


train_steps_per_epoch = x_train.shape[0]//batch_size
val_steps_per_epoch = x_val.shape[0]//batch_size
history = model.fit_generator(train_generator,
                              steps_per_epoch=train_steps_per_epoch,
                              validation_data=val_generator,
                              validation_steps=val_steps_per_epoch,
                              epochs = 30, verbose=1,
                              callbacks=[early_stop])


# In[32]:


#plot accuracy and loss
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


# #### 7.2.2.1 Model Testing & Result Analysis<a name="7221"></a>

# In[33]:


from sklearn.metrics import confusion_matrix
# Evaluate the model on the test set

test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.n//test_generator.batch_size)

# print the test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
# Evaluate the model on the test set

# make predictions on test data
y_pred_proba = model.predict(test_generator)
# convert probabilities to class labels
y_pred = np.argmax(y_pred_proba, axis=1)

# get true labels
y_true = np.argmax(y_test, axis=1)


print(classification_report(y_true,y_pred ))

# create confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# plot the confusion matrix
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Dog Breed Classification VGG16')
plt.colorbar()
tick_marks = np.arange(len(breed_to_num))
plt.xticks(tick_marks, breed_to_num.keys(), rotation=90)
plt.yticks(tick_marks, breed_to_num.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# add the counts to the plot
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.show()


# In[34]:


# switch the keys and values of the original dictionary
num_to_breed = {v: k for k, v in breed_to_num.items()}

# convert numerical labels to breed names for target labels and predictions
target_breeds = [num_to_breed[num] for num in y_true]
prediction_breeds = [num_to_breed[num] for num in y_pred]

predict_df = pd.DataFrame({'Target_Labels':target_breeds, 'predictions': prediction_breeds})
predict_df


# #### 7.2.2.2 Error Analysis<a name="7222"></a>

# In[35]:


plt.figure(figsize = (30,50))
for counter, i in enumerate(random.sample(range(0, len(y_test)), 100)):
    plt.subplot(20, 5, counter+1)
    plt.subplots_adjust(hspace=0.6)
    actual = str(target_breeds[i])
    predicted = str(prediction_breeds[i])
    confidence = str(max(y_pred_proba[i]))
    plt.imshow(X_test[i]/255.0)
    plt.axis('off')
    plt.title('Actual :'+actual+'\nPredict :'+predicted+'\nConfidence :'+confidence, fontsize=12)
plt.show()


# In[36]:


predict_df['count'] = 1
misclass_df = predict_df[predict_df['Target_Labels'] != predict_df['predictions']].groupby(['Target_Labels', 'predictions']).sum().sort_values(['count'], ascending=False).reset_index()

#Create a column to display breed pairs
misclass_df['pair'] = misclass_df['Target_Labels'] + ' / ' + misclass_df['predictions']

#Select the top 30 misclassified breed pairs
misclass_df = misclass_df[['pair', 'count']].head(30)

#Sort the dataframe by count
misclass_df = misclass_df.sort_values(['count'])

#Create horizontal bar plot
fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(misclass_df['pair'], misclass_df['count'], align='center')
ax.invert_yaxis()
ax.set_xlabel('Misclassified Count')
ax.set_ylabel('Breed Pair')
ax.set_title('Top 30 Misclassified Breed Pairs')
plt.show()


# In[37]:


# Analyze misclassified images
misclassified_indices = np.where(y_pred != y_true)[0]
misclassified_images = test_data[misclassified_indices]
misclassified_true_labels = y_true[misclassified_indices]
misclassified_pred_labels = y_pred[misclassified_indices]

for i in range(len(misclassified_indices)):
    image = misclassified_images[i]
    true_label = misclassified_true_labels[i]
    pred_label = misclassified_pred_labels[i]
    plt.imshow(image)
    plt.title(f'True label: {true_label}, Predicted label: {pred_label}')
    plt.show()


# ### 7.3 Xception<a name="73"></a>

# #### 7.3.1 Model with hyperparameters<a name="731"></a>

# In[38]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define input shape
input_shape = (150, 150, 3)

# Load pre-trained Xception model with imagenet weights
xception = Xception(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the pre-trained layers
for layer in xception.layers:
    layer.trainable = False

# Add a custom classification head
custom_head_output = GlobalAveragePooling2D()(xception.output)
custom_head_output = Dense(1024, activation='relu')(custom_head_output)
custom_head_output = Dense(512, activation='relu')(custom_head_output)
custom_head_output = Dropout(0.5)(custom_head_output)
custom_head_output = Dense(10, activation='softmax')(custom_head_output)

# Define the model
model = Model(inputs=xception.input, outputs=custom_head_output)

# Compile the model with categorical cross-entropy loss and Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Define early stopping callback
early_stop = EarlyStopping(patience=5, monitor='val_loss')


# In[39]:


model.summary()


# In[32]:


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# In[40]:


# Train the model with early stopping
history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.n//batch_size,
                              validation_data=val_generator,
                              validation_steps=validation_steps,
                              epochs=30, verbose=1,
                              callbacks=[early_stop])


# In[41]:


#plot accuracy and loss
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


# #### 7.3.1.1 Model Testing & Result Analysis<a name="7311"></a>

# In[42]:


from sklearn.metrics import confusion_matrix
# Evaluate the model on the test set

test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.n//test_generator.batch_size)

# print the test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
# Evaluate the model on the test set

# make predictions on test data
y_pred_proba = model.predict(test_generator)
# convert probabilities to class labels
y_pred = np.argmax(y_pred_proba, axis=1)

# get true labels
y_true = np.argmax(y_test, axis=1)


print(classification_report(y_true,y_pred ))

# create confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# plot the confusion matrix
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Dog Breed Classification Xception')
plt.colorbar()
tick_marks = np.arange(len(breed_to_num))
plt.xticks(tick_marks, breed_to_num.keys(), rotation=90)
plt.yticks(tick_marks, breed_to_num.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# add the counts to the plot
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.show()


# In[43]:


# switch the keys and values of the original dictionary
num_to_breed = {v: k for k, v in breed_to_num.items()}

# convert numerical labels to breed names for target labels and predictions
target_breeds = [num_to_breed[num] for num in y_true]
prediction_breeds = [num_to_breed[num] for num in y_pred]

predict_df = pd.DataFrame({'Target_Labels':target_breeds, 'predictions': prediction_breeds})
predict_df


# #### 7.3.1.2 Error Analysis<a name="7312"></a>

# In[44]:


predict_df['count'] = 1
misclass_df = predict_df[predict_df['Target_Labels'] != predict_df['predictions']].groupby(['Target_Labels', 'predictions']).sum().sort_values(['count'], ascending=False).reset_index()

#Create a column to display breed pairs
misclass_df['pair'] = misclass_df['Target_Labels'] + ' / ' + misclass_df['predictions']

#Select the top 30 misclassified breed pairs
misclass_df = misclass_df[['pair', 'count']].head(30)

#Sort the dataframe by count
misclass_df = misclass_df.sort_values(['count'])

#Create horizontal bar plot
fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(misclass_df['pair'], misclass_df['count'], align='center')
ax.invert_yaxis()
ax.set_xlabel('Misclassified Count')
ax.set_ylabel('Breed Pair')
ax.set_title('Top 30 Misclassified Breed Pairs')
plt.show()


# In[45]:


plt.figure(figsize = (30,50))
for counter, i in enumerate(random.sample(range(0, len(y_test)), 100)):
    plt.subplot(20, 5, counter+1)
    plt.subplots_adjust(hspace=0.6)
    actual = str(target_breeds[i])
    predicted = str(prediction_breeds[i])
    confidence = str(max(y_pred_proba[i]))
    plt.imshow(X_test[i]/255.0)
    plt.axis('off')
    plt.title('Actual :'+actual+'\nPredict :'+predicted+'\nConfidence :'+confidence, fontsize=12)
plt.show()


# In[46]:


# Analyze misclassified images
misclassified_indices = np.where(y_pred != y_true)[0]
misclassified_images = test_data[misclassified_indices]
misclassified_true_labels = y_true[misclassified_indices]
misclassified_pred_labels = y_pred[misclassified_indices]

for i in range(len(misclassified_indices)):
    image = misclassified_images[i]
    true_label = misclassified_true_labels[i]
    pred_label = misclassified_pred_labels[i]
    plt.imshow(image)
    plt.title(f'True label: {true_label}, Predicted label: {pred_label}')
    plt.show()


# ### 7.4 InceptionResNetV2<a name="74"></a>

# #### 7.4.1 Model with hyperparameters<a name="741"></a>

# In[47]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define input shape
input_shape = (150, 150, 3)

# Load pre-trained InceptionResNetV2 model with imagenet weights
inception_resnet_v2 = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the pre-trained layers
for layer in inception_resnet_v2.layers:
    layer.trainable = False

# Add a custom classification head
custom_head_output = Flatten()(inception_resnet_v2.output)
custom_head_output = Dense(256, activation='relu')(custom_head_output)
custom_head_output = Dropout(0.5)(custom_head_output)
custom_head_output = Dense(10, activation='softmax')(custom_head_output)

# Define the model
model = Model(inputs=inception_resnet_v2.input, outputs=custom_head_output)

# Compile the model with categorical cross-entropy loss and Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping callback
early_stop = EarlyStopping(patience=5, monitor='val_loss')


# In[48]:


print(model.summary())


# In[34]:


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# In[49]:


history = model.fit_generator(train_generator,
                              steps_per_epoch=train_steps_per_epoch,
                              validation_data=val_generator,
                              validation_steps=val_steps_per_epoch,
                              epochs=30,
                              verbose=1,
                              callbacks=[early_stop])


# In[50]:


# Evaluate the model on the test set

test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.n//test_generator.batch_size)

# print the test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


# In[51]:


#plot accuracy and loss
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


# #### 7.4.1.1 Model Testing & Result Analysis<a name="7411"></a>

# In[52]:


from sklearn.metrics import confusion_matrix
# Evaluate the model on the test set

test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.n//test_generator.batch_size)

# print the test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
# Evaluate the model on the test set

# make predictions on test data
y_pred_proba = model.predict(test_generator)
# convert probabilities to class labels
y_pred = np.argmax(y_pred_proba, axis=1)

# get true labels
y_true = np.argmax(y_test, axis=1)


print(classification_report(y_true,y_pred ))

# create confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# plot the confusion matrix
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Dog Breed Classification InceptionResNetV2')
plt.colorbar()
tick_marks = np.arange(len(breed_to_num))
plt.xticks(tick_marks, breed_to_num.keys(), rotation=90)
plt.yticks(tick_marks, breed_to_num.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# add the counts to the plot
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.show()


# In[53]:


# switch the keys and values of the original dictionary
num_to_breed = {v: k for k, v in breed_to_num.items()}

# convert numerical labels to breed names for target labels and predictions
target_breeds = [num_to_breed[num] for num in y_true]
prediction_breeds = [num_to_breed[num] for num in y_pred]

predict_df = pd.DataFrame({'Target_Labels':target_breeds, 'predictions': prediction_breeds})
predict_df


# #### 7.4.1.2 Error Analysis<a name="7412"></a>

# In[54]:


plt.figure(figsize = (30,50))
for counter, i in enumerate(random.sample(range(0, len(y_test)), 100)):
    plt.subplot(20, 5, counter+1)
    plt.subplots_adjust(hspace=0.6)
    actual = str(target_breeds[i])
    predicted = str(prediction_breeds[i])
    confidence = str(max(y_pred_proba[i]))
    plt.imshow(X_test[i]/255.0)
    plt.axis('off')
    plt.title('Actual :'+actual+'\nPredict :'+predicted+'\nConfidence :'+confidence, fontsize=12)
plt.show()


# In[55]:


predict_df['count'] = 1
misclass_df = predict_df[predict_df['Target_Labels'] != predict_df['predictions']].groupby(['Target_Labels', 'predictions']).sum().sort_values(['count'], ascending=False).reset_index()

#Create a column to display breed pairs
misclass_df['pair'] = misclass_df['Target_Labels'] + ' / ' + misclass_df['predictions']

#Select the top 30 misclassified breed pairs
misclass_df = misclass_df[['pair', 'count']].head(30)

#Sort the dataframe by count
misclass_df = misclass_df.sort_values(['count'])

#Create horizontal bar plot
fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(misclass_df['pair'], misclass_df['count'], align='center')
ax.invert_yaxis()
ax.set_xlabel('Misclassified Count')
ax.set_ylabel('Breed Pair')
ax.set_title('Top 30 Misclassified Breed Pairs')
plt.show()


# In[56]:


# Analyze misclassified images
misclassified_indices = np.where(y_pred != y_true)[0]
misclassified_images = test_data[misclassified_indices]
misclassified_true_labels = y_true[misclassified_indices]
misclassified_pred_labels = y_pred[misclassified_indices]

for i in range(len(misclassified_indices)):
    image = misclassified_images[i]
    true_label = misclassified_true_labels[i]
    pred_label = misclassified_pred_labels[i]
    plt.imshow(image)
    plt.title(f'True label: {true_label}, Predicted label: {pred_label}')
    plt.show()


# ### 7.5 Inception V3 model<a name="75"></a>

# In[57]:


# Data Augmentation
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

batch_size = 32


train_datagen = ImageDataGenerator(rescale=1./255.,
    rotation_range = 30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow(x_train, y_train, shuffle = False, batch_size = batch_size, seed = 1)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,)

val_generator = val_datagen.flow(x_val, y_val, shuffle = False, batch_size = batch_size, seed = 1)


# In[58]:


img_id = 2
dog_generator = train_datagen.flow(x_train[img_id:img_id+1], y_train[img_id:img_id+1],
                                  shuffle = False, batch_size = batch_size, seed = 1)

plt.figure(figsize=(20,20))
dogs = [next(dog_generator) for i in range(0,5)]
for counter, dog in enumerate(dogs):
    plt.subplot(1,5, counter+1)
    plt.imshow(dog[0][0])
plt.show()


# #### 7.5.1 Model with hyperparameters<a name="751"></a>

# In[59]:


# Define the base model with pre-trained weights
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(150,150,3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add new trainable layers on top of the frozen layers
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(300, activation='relu'))
model.add(Dense(len(breeds), activation='softmax'))

#Freeze pre-trained layers
print('Number of trainable weights before freezing the base layer:', len(model.trainable_weights))
model.layers[0].trainable = False
print('Number of trainable weights after freezing the base layer:', len(model.trainable_weights))


# In[60]:


model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[36]:


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# In[61]:


train_steps_per_epoch = x_train.shape[0] // batch_size
val_steps_per_epoch = x_val.shape[0] // batch_size

# Define early stopping callback
early_stop = EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_steps_per_epoch,
                              validation_data=val_generator,
                              validation_steps=val_steps_per_epoch,
                              epochs=30,
                              verbose=1,
                              callbacks=[early_stop])


# In[62]:


#plot accuracy and loss
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


# #### 7.5.1.1 Model Testing & Result Analysis<a name="7511"></a>

# In[63]:


from sklearn.metrics import confusion_matrix
# Evaluate the model on the test set

test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.n//test_generator.batch_size)

# print the test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
# Evaluate the model on the test set

# make predictions on test data
y_pred_proba = model.predict(test_generator)
# convert probabilities to class labels
y_pred = np.argmax(y_pred_proba, axis=1)

# get true labels
y_true = np.argmax(y_test, axis=1)


print(classification_report(y_true,y_pred ))

# create confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# plot the confusion matrix
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Dog Breed Classification Inception V3')
plt.colorbar()
tick_marks = np.arange(len(breed_to_num))
plt.xticks(tick_marks, breed_to_num.keys(), rotation=90)
plt.yticks(tick_marks, breed_to_num.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# add the counts to the plot
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.show()


# In[64]:


# switch the keys and values of the original dictionary
num_to_breed = {v: k for k, v in breed_to_num.items()}

# convert numerical labels to breed names for target labels and predictions
target_breeds = [num_to_breed[num] for num in y_true]
prediction_breeds = [num_to_breed[num] for num in y_pred]

predict_df = pd.DataFrame({'Target_Labels':target_breeds, 'predictions': prediction_breeds})
predict_df


# #### 7.5.1.2 Error Analysis<a name="7512"></a>

# In[65]:


plt.figure(figsize = (30,50))
for counter, i in enumerate(random.sample(range(0, len(y_test)), 100)):
    plt.subplot(20, 5, counter+1)
    plt.subplots_adjust(hspace=0.6)
    actual = str(target_breeds[i])
    predicted = str(prediction_breeds[i])
    confidence = str(max(y_pred_proba[i]))
    plt.imshow(X_test[i]/255.0)
    plt.axis('off')
    plt.title('Actual :'+actual+'\nPredict :'+predicted+'\nConfidence :'+confidence, fontsize=12)
plt.show()


# In[66]:


predict_df['count'] = 1
misclass_df = predict_df[predict_df['Target_Labels'] != predict_df['predictions']].groupby(['Target_Labels', 'predictions']).sum().sort_values(['count'], ascending=False).reset_index()

#Create a column to display breed pairs
misclass_df['pair'] = misclass_df['Target_Labels'] + ' / ' + misclass_df['predictions']

#Select the top 30 misclassified breed pairs
misclass_df = misclass_df[['pair', 'count']].head(30)

#Sort the dataframe by count
misclass_df = misclass_df.sort_values(['count'])

#Create horizontal bar plot
fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(misclass_df['pair'], misclass_df['count'], align='center')
ax.invert_yaxis()
ax.set_xlabel('Misclassified Count')
ax.set_ylabel('Breed Pair')
ax.set_title('Top 30 Misclassified Breed Pairs')
plt.show()


# In[67]:


# Analyze misclassified images
misclassified_indices = np.where(y_pred != y_true)[0]
misclassified_images = test_data[misclassified_indices]
misclassified_true_labels = y_true[misclassified_indices]
misclassified_pred_labels = y_pred[misclassified_indices]

for i in range(len(misclassified_indices)):
    image = misclassified_images[i]
    true_label = misclassified_true_labels[i]
    pred_label = misclassified_pred_labels[i]
    plt.imshow(image)
    plt.title(f'True label: {true_label}, Predicted label: {pred_label}')
    plt.show()


# ## Xception on whole data set

# In[3]:


# loading the all the images

dog_classes = os.listdir(file_path)
breeds = [breed.split('-',1)[1] for breed in dog_classes]


# In[4]:


# reading the path name, Breed name, label of the each image 

from itertools import chain
x, y, z, label = [], [], [], []
fullpaths = [file_path+'/{}'.format(dog_class) for dog_class in dog_classes]
for counter, fullpath in enumerate(fullpaths):
    for imgname in os.listdir(fullpath):
        x.append([fullpath+'/'+imgname])
        y.append(breeds[counter])
        label.append(imgname.split('.')[0])
x=list(chain.from_iterable(x))


# In[5]:


combined = list(zip(x,label,y))
x[:], label[:], y[:] = zip(*combined)
df = pd.DataFrame(combined, columns =['Path', 'id', 'breed'])
df.head()


# In[6]:


df.shape


# In[7]:


# Getting the Dog breed name
Breeds = df['breed'].unique()


# In[8]:


# Get indices of rows for each breed name
for category in Breeds:
    breed_indices = df.index[df['breed'] == category]

    # Split indices into train, test sets with 70%, 30% respectively
    train_indices, test_indices = train_test_split(breed_indices, test_size=0.3, random_state=42)

    # Create a new column in the dataframe with the label
    df.loc[train_indices, 'label'] = 'train'
    df.loc[test_indices, 'label'] = 'test'
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
df['Target'] = labelencoder.fit_transform(df['breed'])
df


# In[9]:


labels = df[['id','breed']][(df["label"] == 'train')].reset_index(drop=True)
train = df[['Path', 'id','breed']][(df["label"] == 'train')].reset_index(drop=True)
test = df[['Path', 'id','breed']][(df["label"] == 'test')].reset_index(drop=True)

train_x = train.iloc[:, 0]
train_y = train.iloc[:, 1:]


test_x = test.iloc[:, 0]
test_y = test.iloc[:, 1:]
#Create list of alphabetically sorted labels.

classes = sorted(list(set(labels['breed'])))
n_classes = len(classes)
print('Total unique breed {}'.format(n_classes))

#Map each label string to an integer label.
breed_to_num = dict(zip(classes, range(n_classes)))
breed_to_num


# In[13]:


X,y = images_to_array(train_x, train_y)
test_data = images_to_array_test(test_x, img_size)
X_test,y_test = images_to_array(test_x, test_y)


# In[14]:


# assume x_train and y_train are the training data
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the image size and batch size for the data generator
img_size = (150, 150)
batch_size = 32

# Define the data generator for the train, validation, and test sets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Define the data generator for the train and test sets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# define the validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)

# create validation generator
val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size)

# compute the validation steps
validation_steps = val_generator.n // val_generator.batch_size

# Create the data generators for the train and test sets
batch_size = 32
train_generator = train_datagen.flow(
    X,
    y,
    batch_size=batch_size,
    shuffle=True,
    seed=42)

test_generator = test_datagen.flow(
    X_test,
    y_test,
    batch_size=batch_size,
    shuffle=False)


# In[15]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define input shape
input_shape = (150, 150, 3)

# Load pre-trained Xception model with imagenet weights
xception = Xception(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the pre-trained layers
for layer in xception.layers:
    layer.trainable = False

# Add a custom classification head
custom_head_output = GlobalAveragePooling2D()(xception.output)
custom_head_output = Dense(1024, activation='relu')(custom_head_output)
custom_head_output = Dense(512, activation='relu')(custom_head_output)
custom_head_output = Dropout(0.5)(custom_head_output)
custom_head_output = Dense(120, activation='softmax')(custom_head_output)

# Define the model
model = Model(inputs=xception.input, outputs=custom_head_output)

# Compile the model with categorical cross-entropy loss and Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Define early stopping callback
early_stop = EarlyStopping(patience=5, monitor='val_loss')


# while running the code it crashed unable to run after multiple try.
# Hence we are uploading the code as it is.

# In[ ]:


# Train the model with early stopping
history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.n//batch_size,
                              validation_data=val_generator,
                              validation_steps=validation_steps,
                              epochs=50, verbose=1,
                              callbacks=[early_stop])


# In[ ]:


#plot accuracy and loss
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix
# Evaluate the model on the test set

test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.n//test_generator.batch_size)

# print the test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
# Evaluate the model on the test set

# make predictions on test data
y_pred_proba = model.predict(test_generator)
# convert probabilities to class labels
y_pred = np.argmax(y_pred_proba, axis=1)

# get true labels
y_true = np.argmax(y_test, axis=1)


print(classification_report(y_true,y_pred ))

# create confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# plot the confusion matrix
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Dog Breed Classification Xception')
plt.colorbar()
tick_marks = np.arange(len(breed_to_num))
plt.xticks(tick_marks, breed_to_num.keys(), rotation=90)
plt.yticks(tick_marks, breed_to_num.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# add the counts to the plot
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.show()


# In[ ]:


# switch the keys and values of the original dictionary
num_to_breed = {v: k for k, v in breed_to_num.items()}

# convert numerical labels to breed names for target labels and predictions
target_breeds = [num_to_breed[num] for num in y_true]
prediction_breeds = [num_to_breed[num] for num in y_pred]

predict_df = pd.DataFrame({'Target_Labels':target_breeds, 'predictions': prediction_breeds})
predict_df


# In[ ]:


plt.figure(figsize = (30,50))
for counter, i in enumerate(random.sample(range(0, len(y_test)), 100)):
    plt.subplot(20, 5, counter+1)
    plt.subplots_adjust(hspace=0.6)
    actual = str(target_breeds[i])
    predicted = str(prediction_breeds[i])
    confidence = str(max(y_pred_proba[i]))
    plt.imshow(X_test[i]/255.0)
    plt.axis('off')
    plt.title('Actual :'+actual+'\nPredict :'+predicted+'\nConfidence :'+confidence, fontsize=12)
plt.show()


# In[ ]:


predict_df['count'] = 1
misclass_df = predict_df[predict_df['Target_Labels'] != predict_df['predictions']].groupby(['Target_Labels', 'predictions']).sum().sort_values(['count'], ascending=False).reset_index()

#Create a column to display breed pairs
misclass_df['pair'] = misclass_df['Target_Labels'] + ' / ' + misclass_df['predictions']

#Select the top 30 misclassified breed pairs
misclass_df = misclass_df[['pair', 'count']].head(30)

#Sort the dataframe by count
misclass_df = misclass_df.sort_values(['count'])

#Create horizontal bar plot
fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(misclass_df['pair'], misclass_df['count'], align='center')
ax.invert_yaxis()
ax.set_xlabel('Misclassified Count')
ax.set_ylabel('Breed Pair')
ax.set_title('Top 30 Misclassified Breed Pairs')
plt.show()


# ## 7.6 Ensemble of Pretrained Models<a name="76"></a>

# In[ ]:


# loading the all the images

dog_classes = os.listdir(file_path)
breeds = [breed.split('-',1)[1] for breed in dog_classes]


# In[ ]:


# reading the path name, Breed name, label of the each image 

from itertools import chain
x, y, z, label = [], [], [], []
fullpaths = [file_path+'/{}'.format(dog_class) for dog_class in dog_classes]
for counter, fullpath in enumerate(fullpaths):
    for imgname in os.listdir(fullpath):
        x.append([fullpath+'/'+imgname])
        y.append(breeds[counter])
        label.append(imgname.split('.')[0])
x=list(chain.from_iterable(x))


# ### 7.6.1 Loading the Whole dataset<a name="761"></a>

# In[ ]:


combined = list(zip(x,label,y))
x[:], label[:], y[:] = zip(*combined)
df = pd.DataFrame(combined, columns =['Path', 'id', 'breed'])
df.head()


# In[ ]:


df['breed'].value_counts()


# In[ ]:


# Getting the Dog breed name
Breeds = df['breed'].unique()

# Get indices of rows for each breed name
for category in Breeds:
    breed_indices = df.index[df['breed'] == category]

    # Split indices into train, test sets with 70%, 30% respectively
    train_indices, test_indices = train_test_split(breed_indices, test_size=0.3, random_state=42)

    # Create a new column in the dataframe with the label
    df.loc[train_indices, 'label'] = 'train'
    df.loc[test_indices, 'label'] = 'test'
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
df['Target'] = labelencoder.fit_transform(df['breed'])
df


# #### 7.6.2 Analysis of the Dataset<a name="762"></a>

# In[ ]:


labels = df[['id','breed']][(df["label"] == 'train')].reset_index(drop=True)
labels.head()


# In[ ]:


#describe
labels.describe()


# In[ ]:


#function to show bar length
def bar_chart(ax): 
    
    for p in ax.patches:
        val = p.get_width() #height of the bar
        x = p.get_x()+ p.get_width() # x- position 
        y = p.get_y() + p.get_height()/2 #y-position
        ax.annotate(round(val,2),(x,y))
        
#finding top dog brands

plt.figure(figsize = (15,30))
ax0 =sns.countplot(y=labels['breed'],order=labels['breed'].value_counts().index)
bar_chart(ax0)
plt.show()


# In[ ]:



#Create list of alphabetically sorted labels.
classes = sorted(list(set(labels['breed'])))
n_classes = len(classes)
print('Total unique breed {}'.format(n_classes))



#Map each label string to an integer label.
breed_to_num = dict(zip(classes, range(n_classes)))
breed_to_num


# #### 7.6.3 Create a dataframe for train and test seperately<a name="763"></a>

# In[ ]:


train = df[['Path', 'id','breed']][(df["label"] == 'train')].reset_index(drop=True)
test = df[['Path', 'id','breed']][(df["label"] == 'test')].reset_index(drop=True)

train_x = train.iloc[:, 0]
train_y = train.iloc[:, 1:]


test_x = train.iloc[:, 0]
test_y = train.iloc[:, 1:]


# **One-hot Encoding:**
# Because our predictor's output for each input is a vector of probabilities for each class, we must convert our label dataset to the same format. That is, for each input, a num classes-long row vector with a 1 at the index of the label and 0's everywhere else.

# #### 7.6.4 Data Pre-processing<a name="764"></a>

# In[ ]:


input_shape = (150,150,3)

# convert the image to array for train image and the label

def images_to_array(directory, label_dataframe, target_size = input_shape):
    
    image_labels = label_dataframe['breed']
    images = np.zeros([len(label_dataframe), target_size[0], target_size[1], target_size[2]],dtype=np.uint8) #as we have huge data and limited ram memory. uint8 takes less memory
    y = np.zeros([len(label_dataframe),1],dtype = np.uint8)
    
    for ix in range(len(directory)):
        img_dir = os.path.join(directory[ix])
        img = load_img(img_dir, target_size = target_size)
#         img = np.expand_dims(img, axis=0)
#         img = processed_image_resnet(img)
#         img = img/255
        images[ix]=img
#         images[ix] = img_to_array(img)
        del img
        
        dog_breed = image_labels[ix]
        y[ix] = breed_to_num[dog_breed]
    
    y = to_categorical(y)
    
    return images,y


# In[ ]:


import time 
t = time.time()

X,y = images_to_array(train_x, train_y)

print('runtime in seconds: {}'.format(time.time() - t))


# In[ ]:


# np.where(y[5]==1)[0][0]

# lets check some dogs and their breeds
n=25

# setup the figure 
plt.figure(figsize=(20,20))

for i in range(n):
#     print(i)
    ax = plt.subplot(5, 5, i+1)
    plt.title(classes[np.where(y[i] ==1)[0][0]])
    plt.imshow(X[i].astype('int32')) # .astype('int32') ---> as imshow() needs integer data to read the image
    


# ### Creating callbacks:
# 
# (things to help our model)
# 
# Callbacks are helper functions a model can use during training to do things such as save a models progress, check a models progress or stop training early if a model stops improving.
# 
# ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving.

# In[ ]:


#Learning Rate Annealer
lrr= ReduceLROnPlateau(monitor='val_acc', factor=.01, patience=3, min_lr=1e-5,verbose = 1)

#Prepare call backs
EarlyStop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Hyperparameters
batch_size= 128
epochs=50
learn_rate=.001
sgd=SGD(lr=learn_rate,momentum=.9,nesterov=False)
adam=Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None,  amsgrad=False)


# #### 7.6.5 Feature Extraction<a name="765"></a>

# In[ ]:


#function to extract features from the dataset by a given pretrained model
img_size = (150,150,3)

def get_features(model_name, model_preprocessor, input_size, data):

    input_layer = Input(input_size)
    preprocessor = Lambda(model_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs = input_layer, outputs = avg)
    
    #Extract feature.
    feature_maps = feature_extractor.predict(data, verbose=1)
    print('Feature maps shape: ', feature_maps.shape)
    return feature_maps


# In[ ]:


# Extract features using InceptionV3 
from keras.applications.inception_v3 import InceptionV3, preprocess_input
inception_preprocessor = preprocess_input
inception_features = get_features(InceptionV3,
                                  inception_preprocessor,
                                  img_size, X)


# In[ ]:


# Extract features using Xception 
from keras.applications.xception import Xception, preprocess_input
xception_preprocessor = preprocess_input
xception_features = get_features(Xception,
                                 xception_preprocessor,
                                 img_size, X)


# In[ ]:


# Extract features using InceptionResNetV2 
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
inc_resnet_preprocessor = preprocess_input
inc_resnet_features = get_features(InceptionResNetV2,
                                   inc_resnet_preprocessor,
                                   img_size, X)


# In[ ]:


# Extract features using NASNetLarge 
from keras.applications.nasnet import NASNetLarge, preprocess_input
nasnet_preprocessor = preprocess_input
nasnet_features = get_features(NASNetLarge,
                               nasnet_preprocessor,
                               img_size, X)


# In[ ]:


del X #to free up some ram memory
gc.collect()


# In[ ]:


#Creating final featuremap by combining all extracted features

final_features = np.concatenate([inception_features,
                                 xception_features,
                                 nasnet_features,
                                 inc_resnet_features,], axis=-1) #axis=-1 to concatinate horizontally

print('Final feature maps shape', final_features.shape)


# #### 7.6.6 Model Creation<a name="766"></a>

# In[ ]:


#Prepare Deep net

model = Sequential()
model.add(Dropout(0.7,input_shape=(final_features.shape[1],)))
model.add(Dense(n_classes,activation= 'softmax'))

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


print(model.summary())


# In[ ]:


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# In[ ]:


#Training the model. 
history = model.fit(final_features, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[lrr,EarlyStop])


# In[ ]:


#deleting to free up ram memory

del inception_features
del xception_features
del nasnet_features
del inc_resnet_features
del final_features
gc.collect()


# #### 7.6.6.1 Test Feature Extraction<a name="7661"></a>

# In[ ]:


#Function to read images from test directory

def images_to_array_test(test_path, img_size = (150,150,3)):
    
    data_size = len(test_path)
    images = np.zeros([data_size, img_size[0], img_size[1], 3], dtype=np.uint8)
    
    
    for ix in range(data_size):
        img_dir = os.path.join(test_path[ix])
        img = load_img(img_dir, target_size = img_size)
#         img = np.expand_dims(img, axis=0)
#         img = processed_image_resnet(img)
#         img = img/255
        images[ix]=img
#         images[ix] = img_to_array(img)
        del img
    print('Ouptut Data Size: ', images.shape)
    return images

test_data = images_to_array_test(test_x, img_size)


# In[ ]:


#Extract test data features.
def extact_features(data):
    inception_features = get_features(InceptionV3, inception_preprocessor, img_size, data)
    xception_features = get_features(Xception, xception_preprocessor, img_size, data)
    nasnet_features = get_features(NASNetLarge, nasnet_preprocessor, img_size, data)
    inc_resnet_features = get_features(InceptionResNetV2, inc_resnet_preprocessor, img_size, data)

    final_features = np.concatenate([inception_features,
                                     xception_features,
                                     nasnet_features,
                                     inc_resnet_features],axis=-1)
    
    print('Final feature maps shape', final_features.shape)
    
    #deleting to free up ram memory
    del inception_features
    del xception_features
    del nasnet_features
    del inc_resnet_features
    gc.collect()
    
    
    return final_features

test_features = extact_features(test_data)


# In[ ]:


#Free up some space.
del test_data
gc.collect()
#Predict test labels given test data features.

pred = model.predict(test_features)


# In[ ]:


TEST_xx,TEST_yy = images_to_array(test_x, test_y)


# #### 7.6.6.2 Model Testing & Result Analysis<a name="7662"></a>

# In[ ]:


from sklearn.metrics import confusion_matrix
# Evaluate the model on the test set

# make predictions on test data
y_pred_proba = model.predict(test_features)
# convert probabilities to class labels
y_pred = np.argmax(y_pred_proba, axis=1)

# get true labels
y_true = np.argmax(TEST_yy, axis=1)


print(classification_report(y_true,y_pred ))


# In[ ]:


# Evaluate the model on the test set

test_loss, test_acc = model.evaluate_generator(test_features)

# print the test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


# In[ ]:



# create confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# plot the confusion matrix
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Dog Breed Classification Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(breed_to_num))
plt.xticks(tick_marks, breed_to_num.keys(), rotation=90)
plt.yticks(tick_marks, breed_to_num.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# add the counts to the plot
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.show()


# In[ ]:


# switch the keys and values of the original dictionary
num_to_breed = {v: k for k, v in breed_to_num.items()}

# convert numerical labels to breed names for target labels and predictions
target_breeds = [num_to_breed[num] for num in y_true]
prediction_breeds = [num_to_breed[num] for num in y_pred]

predict_df = pd.DataFrame({'Target_Labels':target_breeds, 'predictions': prediction_breeds})
predict_df


# #### 7.6.6.3 Error Analysis<a name="7663"></a>

# In[ ]:


plt.figure(figsize = (30,50))
for counter, i in enumerate(random.sample(range(0, len(TEST_xx)), 100)):
    plt.subplot(20, 5, counter+1)
    plt.subplots_adjust(hspace=0.6)
    actual = str(target_breeds[i])
    predicted = str(prediction_breeds[i])
    confidence = str(max(y_pred_proba[i]))
    plt.imshow(TEST_xx[i]/255.0)
    plt.axis('off')
    plt.title('Actual :'+actual+'\nPredict :'+predicted+'\nConfidence :'+confidence, fontsize=12)
plt.show()


# In[ ]:


# First prediction
print(pred[0])
print(f"Max value (probability of prediction): {np.max(pred[0])}") # the max probability value predicted by the model
print(f"Sum: {np.sum(pred[0])}") # because we used softmax activation in our model, this will be close to 1
print(f"Max index: {np.argmax(pred[0])}") # the index of where the max value in predictions[0] occurs
print(f"Predicted label: {classes[np.argmax(pred[0])]}")


# ## 8. Testing with custom input:<a name="8"></a>

# In[ ]:


plt.imshow(mpimg.imread(df['Path'][10168]))


# In[ ]:


#reading the image and converting it into an np array

img_g = load_img(df['Path'][10168],target_size = img_size)
img_g = np.expand_dims(img_g, axis=0) # as we trained our model in (row, img_height, img_width, img_rgb) format, np.expand_dims convert the image into this format
# img_g


# In[ ]:


# #Predict test labels given test data features.
test_features = extact_features(img_g)
predg = model.predict(test_features)
print(f"Predicted label: {classes[np.argmax(predg[0])]}")
print(f"Probability of prediction): {round(np.max(predg[0])) * 100} %")

