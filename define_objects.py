from keras.preprocessing import image
from keras.models import Model
from keras.applications.resnet50 import preprocess_input
import numpy as np
import tensorflow as tf
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.models import load_model

   

class DefineObjects:
    global model
    global label_names
    model = load_model('/home/sinem/bitirme/my_stl_model_2.h5')
    
    def __init__(self):
        self.model = load_model('/home/sinem/bitirme/my_stl_model_2.h5')
        self.graph = tf.get_default_graph()

    def extract(self, img): 
       
        x = image.img_to_array(img) 
        x = np.expand_dims(x, axis=0)  
        x /= 255.
        


    def load_label_names():
        with open('/home/sinem/bitirme/class_names.txt', 'r') as f:
           return np.array([l.strip() for l in f])


    label_names = load_label_names()


    def model_predict(self,img):
        img = img.resize((96, 96))
        
        img = img.convert('RGB') 


        x = image.img_to_array(img)  
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x) 
       
        with self.graph.as_default():
            predictions = model.predict(x/255.0)[0]
            predictions *= 100
            
            order = (-predictions).argsort()

            sorted_predictions = list(zip(label_names[order], predictions[order]))
            
           
            return sorted_predictions[:1]
            
