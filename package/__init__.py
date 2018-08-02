#open http://127.0.0.1:5000/ on your browser to see result

from flask import Flask
from keras.applications import xception
from keras.applications import inception_v3
import pickle

#extract bottleneck
inception_bottleneck = inception_v3.InceptionV3(weights='./package/models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, pooling='avg')
inception_bottleneck._make_predict_function()
xception_bottleneck = xception.Xception(weights='./package/models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, pooling='avg')
xception_bottleneck._make_predict_function()
#load model
logreg = pickle.load(open('./package/models/logreg_model.sav', 'rb'))

UPLOAD_FOLDER = './package/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg']) #permitted file extensions for user upload

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import package.upload_img
import package.process_and_report