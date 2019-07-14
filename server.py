import os
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from define_objects import DefineObjects
import glob
import pickle
from datetime import datetime
from flask import Flask, request, render_template
from keras.preprocessing import image
app = Flask(__name__)


# Read image features
fe = FeatureExtractor()
de = DefineObjects()

features = []
predicts = []
img_paths = []

features_2 = []
predicts_2 = []
img_paths_2 = []
sorgu_img_path = ""
scoress=""



ids_box = 15
for feature_path in glob.glob("static/feature/*"):
    features.append(pickle.load(open(feature_path, 'rb')))
    img_paths.append('static/img/' + os.path.splitext(os.path.basename(feature_path))[0] + '.png')

for feature_path in glob.glob("static/feature/*"):
    features_2.append(pickle.load(open(feature_path, 'rb')))
    img_paths_2.append('static/img/' + os.path.splitext(os.path.basename(feature_path))[0] + '.png')


@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        file = request.files['query_img']
        
        img = image.load_img(file.stream,target_size=(96, 96))  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat() + "_" + file.filename
        img.save(uploaded_img_path)

        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)  # Do search
        ids = np.argsort(dists)[:15] 
        scores = [(dists[id], img_paths[id]) for id in ids]

        predicts = de.model_predict(img)

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores,
                               predicts=predicts)



    if request.method == 'GET':
        sorgu = request.args.get('query_text')
        
        if sorgu =='kopek':
           img_2 = image.load_img("static/img/train_image_png_100.png",target_size=(96, 96))  # PIL image
           query_2 = fe.extract(img_2)
           dists_2 = np.linalg.norm(features_2 - query_2, axis=1)  # Do search
           ids_2 = np.argsort(dists_2)[:15] 
           scores_2 = [(dists_2[id], img_paths_2[id]) for id in ids_2]

           

           return render_template('index.html',
                               scores_2=scores_2)

        if sorgu =='ucak':
           img_2 = image.load_img("static/img/train_image_png_31.png",target_size=(96, 96))  # PIL image
           query_2 = fe.extract(img_2)
           dists_2 = np.linalg.norm(features_2 - query_2, axis=1)  # Do search
           ids_2 = np.argsort(dists_2)[:15] 
           scores_2 = [(dists_2[id], img_paths_2[id]) for id in ids_2]

           

           return render_template('index.html',
                               scores_2=scores_2)


        if sorgu =='kus':
           img_2 = image.load_img("static/img/train_image_png_43.png",target_size=(96, 96))  # PIL image
           query_2 = fe.extract(img_2)
           dists_2 = np.linalg.norm(features_2 - query_2, axis=1)  # Do search
           ids_2 = np.argsort(dists_2)[:15] 
           scores_2 = [(dists_2[id], img_paths_2[id]) for id in ids_2]

           

           return render_template('index.html',
                               scores_2=scores_2)


        if sorgu =='araba':
           img_2 = image.load_img("static/img/train_image_png_20.png",target_size=(96, 96))  # PIL image
           query_2 = fe.extract(img_2)
           dists_2 = np.linalg.norm(features_2 - query_2, axis=1)  # Do search
           ids_2 = np.argsort(dists_2)[:15] 
           scores_2 = [(dists_2[id], img_paths_2[id]) for id in ids_2]

           

           return render_template('index.html',
                               scores_2=scores_2)


        if sorgu =='kedi':
           img_2 = image.load_img("static/img/train_image_png_40.png",target_size=(96, 96))  # PIL image
           query_2 = fe.extract(img_2)
           dists_2 = np.linalg.norm(features_2 - query_2, axis=1)  # Do search
           ids_2 = np.argsort(dists_2)[:15] 
           scores_2 = [(dists_2[id], img_paths_2[id]) for id in ids_2]

           

           return render_template('index.html',
                               scores_2=scores_2) 


        if sorgu =='geyik':
           img_2 = image.load_img("static/img/train_image_png_46.png",target_size=(96, 96))  # PIL image
           query_2 = fe.extract(img_2)
           dists_2 = np.linalg.norm(features_2 - query_2, axis=1)  # Do search
           ids_2 = np.argsort(dists_2)[:15] 
           scores_2 = [(dists_2[id], img_paths_2[id]) for id in ids_2]

           

           return render_template('index.html',
                               scores_2=scores_2) 



        if sorgu =='at':
           img_2 = image.load_img("static/img/train_image_png_51.png",target_size=(96, 96))  # PIL image
           query_2 = fe.extract(img_2)
           dists_2 = np.linalg.norm(features_2 - query_2, axis=1)  # Do search
           ids_2 = np.argsort(dists_2)[:15] 
           scores_2 = [(dists_2[id], img_paths_2[id]) for id in ids_2]

           

           return render_template('index.html',
                               scores_2=scores_2) 



        if sorgu =='maymun':
           img_2 = image.load_img("static/img/train_image_png_78.png",target_size=(96, 96))  # PIL image
           query_2 = fe.extract(img_2)
           dists_2 = np.linalg.norm(features_2 - query_2, axis=1)  # Do search
           ids_2 = np.argsort(dists_2)[:15] 
           scores_2 = [(dists_2[id], img_paths_2[id]) for id in ids_2]

           

           return render_template('index.html',
                               scores_2=scores_2) 


        if sorgu =='gemi':
           img_2 = image.load_img("static/img/train_image_png_79.png",target_size=(96, 96))  # PIL image
           query_2 = fe.extract(img_2)
           dists_2 = np.linalg.norm(features_2 - query_2, axis=1)  # Do search
           ids_2 = np.argsort(dists_2)[:15] 
           scores_2 = [(dists_2[id], img_paths_2[id]) for id in ids_2]

           

           return render_template('index.html',
                               scores_2=scores_2) 


        if sorgu =='kamyon':
           img_2 = image.load_img("static/img/train_image_png_80.png",target_size=(96, 96))  # PIL image
           query_2 = fe.extract(img_2)
           dists_2 = np.linalg.norm(features_2 - query_2, axis=1)  # Do search
           ids_2 = np.argsort(dists_2)[:15] 
           scores_2 = [(dists_2[id], img_paths_2[id]) for id in ids_2]

           

           return render_template('index.html',
                               scores_2=scores_2) 

  
        return render_template('index.html')  
  

    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run("0.0.0.0")
