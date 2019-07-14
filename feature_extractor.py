from keras.preprocessing import image
#from keras.applications.resnet50 import preprocess_input
from keras.models import Model
import numpy as np
import tensorflow as tf

import keras
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
from keras import backend as K, regularizers

from keras.models import Sequential
from keras.models import load_model

class FeatureExtractor:
    global model
    model = load_model('/home/sinem/bitirme/my_stl_model_2.h5')
    
    def __init__(self):
        self.model = load_model('/home/sinem/bitirme/my_stl_model_2.h5')
        self.graph = tf.get_default_graph()

    def extract(self, img): 
        img = img.resize((96, 96)) 
        img = img.convert('RGB')  
        x = image.img_to_array(img) 
        x = np.expand_dims(x, axis=0)  
        x = preprocess_input(x)  

        with self.graph.as_default():
            feature = self.model.predict(x/255.0)[0] 
            return feature / np.linalg.norm(feature) 
    
