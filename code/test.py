#!/usr/bin/env python
# coding: utf-8

# In[6]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras import activations
import keras
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPooling2D


# In[7]:


#loading the model

model = keras.models.load_model('/Users/mount/model.h5')


# In[8]:


model.summary()


# In[10]:


model.optimizer


# In[ ]:




