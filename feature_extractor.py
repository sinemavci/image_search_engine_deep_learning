from keras.preprocessing import image
#from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import tensorflow.compat.v1 as tf

import keras
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
from keras import backend as K, regularizers

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

class FeatureExtractor:
    global model
    model = load_model('C:/Users/sinem.avci/Desktop/image_search_engine_deep_learning/my_stl_model_2.h5')
    
    def __init__(self):
        self.model = load_model('C:/Users/sinem.avci/Desktop/image_search_engine_deep_learning/my_stl_model_2.h5')
        self.graph = tf.Graph()

    def extract(self, img): 
        img = img.resize((96, 96)) 
        img = img.convert('RGB')  
        x = image.img_to_array(img) 
        x = np.expand_dims(x, axis=0)  
        x = preprocess_input(x)  
        feature = self.model.predict(x/255.0)[0] 

        return feature / np.linalg.norm(feature)

       # with self.graph.as_default():
             
    
