from flask import send_file
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.applications import xception
from keras.applications import inception_v3
from os.path import join
import matplotlib.pyplot as plt
import io
import os
from package import app, inception_bottleneck, xception_bottleneck, logreg

@app.route('/package/uploaded_file/<filename>')
def uploaded_file(filename):
     #import labels
     NUM_CLASSES = 120
     labels = pd.read_csv('./package/labels.csv')
     selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)

     # to make new predictions, run code below this line
     img_path = join(app.config['UPLOAD_FOLDER'], filename) 
     img = image.load_img(img_path, target_size=(299, 299))
     img = image.img_to_array(img)
     img_prep = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
     imgX = xception_bottleneck.predict(img_prep, batch_size=32, verbose=1)
     imgI  = inception_bottleneck.predict(img_prep, batch_size=32, verbose=1)
     img_stack = np.hstack([imgX, imgI])
     prediction = logreg.predict(img_stack)
    
     #plot image and prediction
     fig, ax = plt.subplots(figsize=(5,5))
     ax.imshow(img / 255.)
     breed = selected_breed_list[int(prediction)]
     ax.text(10, 250, 'Prediction: %s' % breed, color='k', backgroundcolor='g', alpha=0.8)
     ax.axis('off')
     output = io.BytesIO()
     fig.savefig(output)
     output.seek(0)
     if os.path.exists(img_path):
         os.remove(img_path) #delete user input
     return send_file(output, mimetype='image/png') #return the output image to be called in html