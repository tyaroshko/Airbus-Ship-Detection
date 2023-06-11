# Import the necessary libraries
import os
import math

import numpy as np
import pandas as pd
from skimage.io import imread
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split

class DataGenerator(Sequence):
    def __init__(self, images_set, masks_set, batch_size):
        self.images, self.masks = images_set, masks_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.images) / self.batch_size)

    def __getitem__(self, idx):
        batch_images = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        return (
                np.array([
                    imread(os.path.join(TRAIN_FOLDER, img_name)) 
                    for img_name in batch_images]), 
                np.array([
                    masks_as_image(self.masks[self.masks['ImageId'] == img_name]['EncodedPixels']) 
                    for img_name in batch_images]))
        
# Define the constants
TRAIN_FOLDER = "./airbus_ship_detection/train_v2/"
TEST_FOLDER = "./airbus_ship_detection/test_v2/"
CSV_PATH = "./airbus_ship_detection/train_ship_segmentations_v2.csv"

IMG_SIZE = (768, 768, 3)
NUM_CLASSES = 2
BATCH_SIZE = 8
MAX_TRAIN_STEPS = 100
VALIDATION_SPLIT = 0.2
NB_EPOCHS = 5
RANDOM_STATE = 42
VAL_IMAGES = 500

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

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    Convert run-length encoded mask to a binary mask
    
    Args:
        mask_rle (str): Run-length encoded mask string
        shape (tuple): Shape of the output binary mask
    
    Returns:
        numpy.ndarray: Binary mask array
    '''
    # Split the run-length encoded string
    s = mask_rle.split()
    starts = np.asarray(s[0:][::2], dtype=int) - 1
    lengths = np.asarray(s[1:][::2], dtype=int)
    ends = starts + lengths
    
    # Initialize an array for the binary mask
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    # Set the pixels corresponding to the mask region to 1
    for start, end in zip(starts, ends):
        img[start:end] = 1
    
    # Reshape the array to the desired shape
    return img.reshape(shape).T


def masks_as_image(in_mask_list):
    '''
    Combine individual ship masks into a single mask array
    
    Args:
        in_mask_list (list): List of ship masks (run-length encoded strings)
    
    Returns:
        numpy.ndarray: Combined mask array
    '''
    # Initialize an array to hold the combined mask
    all_masks = np.zeros((768, 768), dtype=np.int16)
    
    # Iterate over the ship masks and add them to the combined mask
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    
    # Expand dimensions to match the expected shape
    return np.expand_dims(all_masks, -1)

df = pd.read_csv(CSV_PATH)

def unet(input_shape):
    # Input layer
    inputs = Input(input_shape)

    # Contracting path (left side of the U-Net)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottom of the U-Net
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Expanding path (right side of the U-Net)
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = Conv2D(256, 2, activation='relu', padding='same')(up5)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(128, 2, activation='relu', padding='same')(up6)
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(64, 2, activation='relu', padding='same')(up7)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    # Output layer
    output = Conv2D(1, 1, activation='sigmoid')(conv7)

    # Create the model
    model = Model(inputs=inputs, outputs=output)

    return model


# Create the U-Net model
model = unet(IMG_SIZE)


# Prepering the data for training
df = pd.read_csv(CSV_PATH)
ship_df = df.copy()
ship_df['NumberOfShips'] = ship_df['EncodedPixels'].notnull().astype(int)
ship_df['EncodedPixels'] = ship_df['EncodedPixels'].replace(0, '')
ship_df = ship_df.groupby('ImageId').sum().reset_index()
ship_df["EncodedPixels"] = ship_df["EncodedPixels"].apply(lambda x: x if x != 0 else "")
df = df.fillna("")


def undersample_zeros(df):
    zeros = df[df['NumberOfShips'] == 0].sample(n=25_000, random_state = RANDOM_STATE)
    nonzeros = df[df['NumberOfShips'] != 0]
    return pd.concat((nonzeros, zeros))


train_ships, valid_ships = train_test_split(ship_df, 
                 test_size = 0.3, 
                 stratify = ship_df['NumberOfShips'])
train_ships = undersample_zeros(train_ships)
valid_ships = undersample_zeros(valid_ships)

# Generator for training data
train_gen = DataGenerator(np.array(train_ships['ImageId']), df, BATCH_SIZE)
valid_gen = DataGenerator(np.array(valid_ships['ImageId']), df, BATCH_SIZE)


# Custom metrics
def dice_coef(y_true, y_pred, smooth=1):
    # Reshape the true masks
    y_true = K.cast(y_true, 'float32')
    # Calculate the intersection between predicted and true masks
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    # Calculate the union of predicted and true masks
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    # Calculate the Dice coefficient
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_p_bce(in_gt, in_pred):
    # Combine binary cross-entropy and negative Dice coefficient
    return 1e-3 * binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)

def true_positive_rate(y_true, y_pred):
    # Calculate the true positive rate
    return K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred))) / K.sum(y_true)

def precision(y_true, y_pred):
    # Calculate the precision rate (the proportion of true positive predictions
    # out of all positive predictions)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pred):
    # Calculate the recall rate (the proportion of true positive predictions 
    # out of all positive samples)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    total_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (total_positives + K.epsilon())

def specificity(y_true, y_pred):
    # Calculate the specificity rate (the proportion of true negative 
    # predictions out of all negative samples)
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    total_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (total_negatives + K.epsilon())

def f1_score(y_true, y_pred):
    # Calculate the F1 score (harmonic mean of precision and recall)
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2 * ((precision_val * recall_val) / (precision_val + recall_val + K.epsilon()))


# Compile the model with the Adam optimizer,
# binary cross-entropy and the custom metrics
model.compile(optimizer="adam", 
              loss=dice_p_bce, 
              metrics=[dice_coef, 
                       'binary_accuracy', 
                       true_positive_rate, 
                       precision, 
                       recall, 
                       specificity, 
                       f1_score])

# Path to save intermediate and model weights
weight_path="./models/{}_weights.best.hdf5".format('seg_model')

# Save the model after each epoch if the validation loss improved
checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only = True)

# Reduce the learning rate when the metric has stopped improving
reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, 
                                   patience=3, 
                                   verbose=1, mode='max', min_delta=0.0001, cooldown=2, min_lr=1e-6)

# Stop training when the validation loss has stopped improving
early = EarlyStopping(monitor="val_dice_coef", 
                      mode="max", 
                      patience=15)

# Keep track of training history by creating a callbacks list
callbacks_list = [checkpoint, early, reduceLROnPlat]

history = model.fit(train_gen,
                    validation_data=valid_gen, 
                    epochs=NB_EPOCHS, 
                    callbacks=callbacks_list)

model.load_weights(weight_path)
model.save('models/model.h5')