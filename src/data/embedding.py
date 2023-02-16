import numpy as np
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import cv2


DATASET_PATH = '../data/images/'


def get_image_path(img):
    return DATASET_PATH + img


def load_image(img):
    return cv2.imread(get_image_path(img))


def get_embedding(model, img_name):
    img = image.load_img(get_image_path(img_name), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x).reshape(-1)
