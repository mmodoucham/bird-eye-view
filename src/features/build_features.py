import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.applications.vgg16 import VGG16
from keras.models import Model
import swifter
from data.embedding import get_embedding


DATASET_PATH = '../data/images/'

df = pd.DataFrame({'image': os.listdir(DATASET_PATH)})
df['id'] = df['image'].str.replace('.jpg', '')

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input,
              outputs=base_model.get_layer('fc1').output)

map_embeddings = df['image'].swifter.apply(
    lambda img: get_embedding(model, img))
df_embs = map_embeddings.apply(pd.Series)

np.save('../data/embeddings/embeddings.npy', df_embs)
