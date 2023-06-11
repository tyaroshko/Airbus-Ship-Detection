import os
import numpy as np
import pandas as pd
from skimage.io import imread
import tensorflow.keras as keras
from keras import models, layers


MODEL_PATH = "./models/model.h5"
TEST_FOLDER = "./airbus-ship-detection/test_v2/"

from skimage.morphology import label
def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

model = models.load_model(MODEL_PATH, compile=False)

test_paths = os.listdir(TEST_FOLDER)

out_pred_rows = []
for img_id in test_paths:
    img = imread(os.path.join(TEST_FOLDER, img_id))
    img = np.expand_dims(img, 0)/255.0
    prediction = model.predict(img)[0]
    encodings = multi_rle_encode(prediction)
    # Add an entry with None if there is no ship detected and 
    out_pred_rows.append([{'ImageId': img_id, 'EncodedPixels': encoding} 
                      if encodings 
                      else {'ImageId': img_id, 'EncodedPixels': None} 
                      for encoding in encodings])
    
    
result_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
result_df.to_csv('result.csv', index=False)