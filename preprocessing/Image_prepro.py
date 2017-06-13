from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np

def image_resize(image_path):
    image = load_img(image_path, target_size=(224,224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def image_pre(image):
    preprocess = imagenet_utils.preprocess_input
    image = preprocess(image)
    return image
