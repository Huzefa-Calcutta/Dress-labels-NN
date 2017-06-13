from keras.applications.resnet50 import ResNet50, decode_predictions

def load_model():
    deep_model =  ResNet50(weights='imagenet')
    return deep_model

def load_pred(img,model_name):
    preds = model_name.(img)
    labels = decode_predictions(preds, top = 5)[0]
    return labels

