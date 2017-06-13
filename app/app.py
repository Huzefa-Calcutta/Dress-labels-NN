import os,sys
import flask
import urllib
import json
from keras.applications.resnet50 import ResNet50, decode_predictions
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import Model.Model as mdl
import preprocessing.Image_prepro as pre

app = flask.Flask(__name__)

def start_app():
    app.run(host="0.0.0.0", debug=True, port=5002, processes=True, use_reloader=True)

@app.route('/dress_label', methods=['POST'])
def label():
    inputs = flask.request.json
    img_url = input['url']
    pic_name = 'tmp.jpg'
    urllib.urlretrieve(img_url, 'tmp/%s' % pic_name)
    img = pre.image_resize('tmp/tmp.jpg')
    img = pre.image_pre(img)
    model_dress = mdl.load_model()
    labels = dict(mdl.load_pred(img,model_dress))
    return flask.Response(json.dumps(labels),mimetype='application/json')

if __name__ == '__main__':
    start_app()



