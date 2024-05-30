import tensorflow as t
import splitfolders

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import matplotlib.pyplot as plt

# Define the directories for the training, validation, and test sets
base_dir = "../data"
input_dir = os.path.join(base_dir, "no_split")
output_dir = os.path.join(base_dir, "split")



# # split dataset into train, validation, and test set in 80%, 10%, and 10% respectively.
splitfolders.ratio(input_dir, output= output_dir, seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False)

# assigning the train, test, validation directory to variables
dir_train = os.path.join(output_dir, "train")
dir_test = os.path.join(output_dir, "test")
dir_val = os.path.join(output_dir, "val")


def datagen(batchSize=32, preproc_func=tf.keras.applications.resnet.preprocess_input, pixel=224):
    """
    Implement data loading using ImageDataGenerator & flow_from_directory
    --------------------------------------------------------------------------
    Arguments:
    pixel: the pixel dimensions
    batchSize: the number of set of images loaded at a time
    preproc_func: the type of input preprocessing based on the specific model
    --------------------------------------------------------------------------
    Returns:
    train_gen, val_gen, test_gen -- variables storing train, test, and validation set.
    """
    datagenerator = ImageDataGenerator(preprocessing_function=preproc_func, shear_range=0.2, zoom_range=0.3,
                                       rotation_range=90, width_shift_range=0.2, height_shift_range=0.2,
                                       horizontal_flip=True, vertical_flip=True) 
    
    train_gen = datagenerator.flow_from_directory(dir_train, target_size=(pixel, pixel), class_mode='binary', batch_size=batchSize, shuffle=True) 
    val_gen = datagenerator.flow_from_directory(dir_val, target_size=(pixel, pixel), class_mode='binary', batch_size=batchSize, shuffle=True)
    test_gen = datagenerator.flow_from_directory(dir_test, target_size=(pixel, pixel), class_mode='binary', batch_size=batchSize, shuffle=True)
    
    # Calculate class weights
    classes = train_gen.classes
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(classes), y=classes)
    class_weights_dict = dict(enumerate(class_weights))

    return train_gen, val_gen, test_gen, class_weights_dict

def plot_history(model_history, name):
    plt.plot(model_history.history['accuracy'], label='accuracy')
    plt.plot(model_history.history['val_accuracy'], label='val_accuracy')
    plt.plot(model_history.history['loss'], label='loss')
    plt.plot(model_history.history['val_loss'], label='val_loss')
    plt.title('Model accuracy and loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.legend()
    plt.savefig(f"../plots/{name}.png")

# Classifier model version 2
def createModel_v2(train_set, val_set, class_weights, path_to_save: str = "../models"):
    model_res50 = ResNet50(weights='imagenet', include_top=False, input_shape=train_set.image_shape)
    
    # Create a Sequential model and add the ResNet50 layers
    model = Sequential()
    model.add(model_res50)
    
    # Freeze the ResNet50 layers
    for layer in model.layers:
        layer.trainable = False
    
    # Add new dense layers
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(1, activation="sigmoid"))
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    # Setting callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    my_callbacks = [early_stopping]
    
    # Train the model
    history = model.fit(train_set, validation_data=val_set, epochs=5, callbacks=my_callbacks, class_weight=class_weights)
    model.save(os.path.join(path_to_save, 'modelResNet_50_v2_pre.keras'))
    
    for layer in model.layers:
        layer.trainable = True
    
    history = model.fit(train_set, validation_data=val_set, epochs=8, initial_epoch=5, callbacks=my_callbacks, class_weight=class_weights)
    model.save(os.path.join(path_to_save, 'modelResNet_50_v2_post.keras'))
    
    return model, history

# Generate train, val, and test split
train_gen, val_gen, test_gen, class_weights = datagen(batchSize=4)
print(len(train_gen.classes), sum(train_gen.classes))
# model_res50_v2, Res50hist_v2 = createModel_v2(train_gen, val_gen, class_weights)
# plot_history(Res50hist_v2, name="v2")
