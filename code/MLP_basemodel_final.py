#!/usr/bin/env python
# coding: utf-8

# In[99]:


# TensorFlow and tf.keras
import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd
import os
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

from keras.layers.core import Dense, Dropout, Activation
#from keras.optimizers import SGD

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

print(tf.__version__)


# In[100]:


all_fruit = os.listdir('/Users/mount/Desktop/traindata_Project309/')
print(all_fruit)


# In[101]:


data_label = []

for item in all_fruit:
    all_vege = os.listdir('/Users/mount/Desktop/traindata_Project309/'+ '/' + item)
     
    for fruit in all_vege:
        data_label.append((item, str('/Users/mount/Desktop/traindata_Project309/' + '/' + item) + '/'+ fruit))
        #print(data_label[:1])


# In[102]:


df = pd.DataFrame(data=data_label, columns=['label', 'image'])


# In[103]:


BASE_PATH = "/Users/mount/Desktop/traindata_Project309/"
img_size = 28

images=[]
labels=[]

for i in all_fruit:
    data_path = BASE_PATH + str(i)
    filenames = [i for i in os.listdir(data_path)]
    #print(filenames)
    
    for f in filenames:
        img = cv2.imread(data_path + '/' + f)
        
        img = cv2.resize(img, (img_size, img_size))
        images.append(img)
        labels.append(i)


# In[104]:


images = np.array(images)
images.shape


# In[105]:


images = images.astype('float')/255.0


# In[106]:


y = df['label'].values
#print(y[:5])

lb = LabelEncoder()

y  = lb.fit_transform(y)
#print(y)


from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = y.reshape(len(y), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
y = onehot_encoded


# In[107]:


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

images, y = shuffle(images, y, random_state=1)

x_train, x_test, y_train, y_test = train_test_split( images, y, test_size=0.20, random_state=14)

print(x_train.shape)


# In[108]:


# define the model
import keras
from tensorflow.keras import activations
from tensorflow.keras.layers import MaxPooling2D

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28,3)),

    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.25),
    
    keras.layers.Dense(256, activation="relu"),      
    keras.layers.Dropout(0.25),
    
    keras.layers.Flatten(),
    keras.layers.Dense(3, activation=tf.nn.softmax)
])


# In[109]:


model.summary()


# In[110]:


# compile the model

#import keras.losses.CategoricalCrossentropy()

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss= tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'], )


# In[129]:


# model.fit(x_train, y_train, epochs=50)

history = model.fit(x_train, y_train, epochs=50, 
                    validation_data=(x_test, y_test))


# In[128]:


loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,51)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[113]:


loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs = range(1,51)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[121]:


# Calling `save('my_model')` creates a SavedModel folder `my_model`.
model.save("/Users/mount/my_model.h5")


# In[124]:


new_model = keras.models.load_model('/Users/mount/my_model.h5')


# In[125]:


new_model.summary()


# In[ ]:




