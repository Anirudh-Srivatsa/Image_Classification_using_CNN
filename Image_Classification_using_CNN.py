#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt


# In[ ]:


os.environ['KAGGLE_USERNAME'] = 'Your_username' 
os.environ['KAGGLE_KEY'] = 'Your_Key'  

# Download and extract the dataset
# Download and extract the dataset
get_ipython().system('kaggle competitions download -c dogs-vs-cats ')
get_ipython().system('unzip -q dogs-vs-cats.zip')
get_ipython().system('unzip -q train.zip -d data')
get_ipython().system('unzip -q test1.zip -d data')

# Load the training data filenames and labels
train_filenames = os.listdir('data/train')
train_labels = [int(filename.split('.')[0] == 'dog') for filename in train_filenames]
df_train = pd.DataFrame({'filename': train_filenames, 'label': train_labels})


# In[34]:


import os
import pandas as pd

train_filenames = os.listdir('data/train')
train_labels = [int(filename.startswith('dog')) for filename in train_filenames]
df_t = pd.DataFrame({'filename': train_filenames, 'label': train_labels})

df_t.head()


# In[5]:


df_t.shape


# In[33]:


import matplotlib.pyplot as plt

counts = df_t['label'].value_counts()

plt.bar(['dog', 'cat'], counts, color=['yellow', 'green'])
plt.title('Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()


# In[8]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
# splitting to train & val
df_t, val_df = train_test_split(df_t, test_size=0.2, random_state=100)



df_t['label'] = df_t['label'].astype(str)
val_df['label'] = val_df['label'].astype(str)

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(dataframe=df_t, directory='data/train', 
                                                    x_col='filename', 
                                                    y_col='label', 
                                                    target_size=(224,224), 
                                                    class_mode='binary', 
                                                    batch_size=64, shuffle=False)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_dataframe(dataframe=val_df, directory='data/train', 
                                                x_col='filename', 
                                                y_col='label', 
                                                target_size=(224,224), 
                                                class_mode='binary', 
                                                batch_size=64, shuffle=False)


# In[39]:


from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    # Define the CNN architecture
    cnn = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    
    return model

cnn = create_model()
cnn.summary()


# In[10]:


history = base_cnn.fit(train_generator, epochs=10, validation_data=val_generator)


# In[11]:


val_loss, val_accuracy = base_cnn.evaluate(val_generator)
print("Validation accuracy:", val_accuracy)


# In[12]:


import matplotlib.pyplot as plt

# training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# training and validation accuracy
plt.subplot(1, 2, 2) 
plt.plot(history.history['acc'], label='training accuracy')
plt.plot(history.history['val_acc'], label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()


# In[13]:


x_val, y_val = next(val_generator)
preds = base_cnn.predict(x_val)

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    ax.imshow(x_val[i])
    ax.set_title(f"True: {y_val[i]}, Pred: {preds[i][0]:.2f}")
    ax.axis('off')


# In[40]:


from tensorflow import keras
from tensorflow.keras import layers, regularizers

model = keras.Sequential([
layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), input_shape=(224, 224, 3)),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
layers.MaxPooling2D((2, 2)),
layers.BatchNormalization(),
layers.Dropout(0.5),
layers.Flatten(),
layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
layers.BatchNormalization(),
layers.Dropout(0.5),
layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


model = create_model()
model.summary()


# In[17]:


# Train the model on the training data
history = model.fit(train_generator, epochs=10, validation_data=val_generator)


# In[18]:


# Evaluate the model on the validation data
val_loss, val_acc = model.evaluate(val_generator)
print("Validation accuracy:", val_acc)


# In[20]:


# Plot training and validation loss/accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2) 
plt.plot(history.history['acc'], label='training accuracy')
plt.plot(history.history['val_acc'], label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()


# In[24]:


import matplotlib.pyplot as plt
import numpy as np

# Generate a batch of validation data
x_val, y_val = val_generator.next()

# Make predictions on the validation data
preds = base_cnn.predict(x_val)

# Display a 3x3 grid of images with their true labels and predicted probabilities
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    # Display the image
    ax.imshow(x_val[i])
    ax.axis('off')
    # Set the title with the true label and predicted probability
    true_label = 'Dog' if y_val[i] == 1 else 'Cat'
    pred_prob = preds[i][0]
    pred_label = 'Dog' if pred_prob >= 0.5 else 'Cat'
    ax.set_title(f"True: {true_label}, Pred: {pred_label} ({pred_prob:.2f})")


# In[29]:


IMAGE_SIZE = (224, 224)
batch_size = 32

# Get the list of filenames in the test directory
test_filenames = os.listdir("data/test1")

# Create a DataFrame with the filenames
test_df = pd.DataFrame({'filename': test_filenames})

# Get the number of samples in the test set
nb_samples = test_df.shape[0]

# Create a test data generator with image rescaling
test_datagen = ImageDataGenerator(rescale=1/255)

# Use the test data generator to create a test data iterator
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory="data/test1/",
    x_col="filename",
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)


# In[31]:


import numpy as np

# Predict on the test set
predictions = model.predict(test_generator, steps=np.ceil(nb_samples/batch_size))

# Threshold for class prediction
threshold = 0.5

# Assign labels to test images based on the prediction output
test_df['label'] = np.where(predictions > threshold, 1, 0)


# In[32]:


# Create output dataframe with image ids and predicted labels
output_df = pd.DataFrame({'id': test_df['filename'].str.split('.').str[0], 'label': test_df['label']})

# Save output dataframe to CSV file
output_df.to_csv('predictions.csv', index=False, columns=['id', 'label'])

