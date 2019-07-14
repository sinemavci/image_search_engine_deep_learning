import glob
import os
import pickle
from PIL import Image
from feature_extractor import FeatureExtractor
from define_objects import DefineObjects

fe = FeatureExtractor()
de = DefineObjects()

predicts = []

f = open("index_label.txt","w")


for img_path in sorted(glob.glob('static/img/*.png')):
    print(img_path)
    img = Image.open(img_path) 
    predicts = de.model_predict(img) 
    #f.writelines([str(img_path),"/t",str(predicts),"/n"])
    f.write("%s \n %s \n" % (str(img_path), str(predicts)))
    feature = fe.extract(img)
    feature_path = 'static/feature/' + os.path.splitext(os.path.basename(img_path))[0] + '.pkl'
    pickle.dump(feature, open(feature_path, 'wb'))
    


    
    
    





