#!/usr/bin/env python
# coding: utf-8

# In[48]:


# TensorFlow and tf.keras
import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd
import os
import keras
#from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

from tensorflow import keras
from tensorflow.keras.models import Sequential
import tensorflow as tf

from tensorflow.keras import datasets, layers, models

from keras.layers.core import Dense, Dropout, Activation
#from keras.optimizers import SGD

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

print(tf.__version__)


# In[49]:


all_fruit = os.listdir('/Users/mount/Desktop/traindata_Project309/')


# In[50]:


data_label = []

for item in all_fruit:
    all_vege = os.listdir('/Users/mount/Desktop/traindata_Project309/'+ '/' + item)
     
    for fruit in all_vege:
        data_label.append((item, str('/Users/mount/Desktop/traindata_Project309/' + '/' + item) + '/'+ fruit))
        #print(data_label[:1])


# In[51]:


df = pd.DataFrame(data=data_label, columns=['label', 'image'])


# In[52]:


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


# In[53]:


images = np.array(images)
images.shape


# In[54]:


#images = images.reshape(-1, 32,32, 1)
#test_X = test_X.reshape(-1, 28,28, 1)
images.shape


# In[55]:


images = images.astype('float')/255.0

# # Reshape the images.
# images = np.expand_dims(images, axis=3)
# #test_images = np.expand_dims(test_images, axis=3)
images.shape


# In[56]:


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


# In[57]:


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

images, y = shuffle(images, y, random_state=1)

x_train, x_test, y_train, y_test = train_test_split( images, y, test_size=0.20, random_state=14)

print(x_train.shape)


# In[58]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras import activations

import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPooling2D


# In[59]:


# # define the CNN model here

model=Sequential()

model.add(Conv2D(filters=28,kernel_size=2,padding="same",input_shape=(28,28,3)))
#model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
model.add(MaxPooling2D(pool_size=2))
#model.add(LeakyReLU(alpha=0.1))
model.add(layers.Activation(activations.relu))
model.add(Dropout(0.15))


model.add(Conv2D(filters=64,kernel_size=2,padding="same"))
model.add(MaxPooling2D(pool_size=2))
#model.add(LeakyReLU(alpha=0.1))
model.add(layers.Activation(activations.relu))
model.add(Dropout(0.15))


model.add(Conv2D(filters=128,kernel_size=2,padding="same"))
model.add(MaxPooling2D(pool_size=2))
#model.add(LeakyReLU(alpha=0.1))
model.add(layers.Activation(activations.relu))
model.add(Dropout(0.15))


model.add(Flatten())
#model.add(Dense(128,activation="linear"))
#model.add(layers.Activation(activations.relu))
#model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.3))
model.add(Dense(3,activation="softmax"))


# In[60]:


model.summary()


# In[63]:


model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss= tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=50, 
                    validation_data=(x_test, y_test))



# In[64]:


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


# In[65]:


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


# In[66]:


# Calling `save('my_model')` creates a SavedModel folder `my_model`.
model.save("/Users/mount/my_model.h5")


# In[67]:


new_model = keras.models.load_model('/Users/mount/my_model.h5')


# In[68]:


new_model.summary()


# In[ ]:




