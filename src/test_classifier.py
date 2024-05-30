import tensorflow as tf
import os 
import shutil
import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Define the directories for the test set
dir_test = '../data/split/test'

def datagen(batchSize=32, pixel=224):
    """
    Implement data loading using ImageDataGenerator & flow_from_directory
    --------------------------------------------------------------------------
    Arguments:
    pixel: the pixel dimensions
    batchSize: the number of set of images loaded at a time
    --------------------------------------------------------------------------
    Returns:
    test_gen -- variable storing test set.
    """
    datagenerator = ImageDataGenerator()
    
    test_gen = datagenerator.flow_from_directory(dir_test, target_size=(pixel, pixel), class_mode='binary', batch_size=batchSize, shuffle=False)
    return test_gen

# Generate the test set
test_gen = datagen(batchSize=4)

# Load the pre-trained model
model_pre = tf.keras.models.load_model('../models/modelResNet_50_v2_pre.keras')
# Load the fully trained model
model_post = tf.keras.models.load_model('../models/modelResNet_50_v2_post.keras')

# Function to evaluate model
def evaluate_model(model, test_gen, output_dir):
    y_pred_prob = model.predict(test_gen)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = test_gen.classes

    filenames = test_gen.filenames

    output_dir_fp = os.path.join(output_dir, "false_positives")
    output_dir_fn = os.path.join(output_dir, "false_negatives")

    # Ensure the output directories exist
    os.makedirs(output_dir_fp, exist_ok=True)
    os.makedirs(output_dir_fn, exist_ok=True)

    # Identify misclassified images and copy them
    for i in range(len(y_true)):
        if y_pred[i] != y_true[i]:
            src = os.path.join(dir_test, filenames[i])
            if y_pred[i] == 1 and y_true[i] == 0:
                dst = os.path.join(output_dir_fp, os.path.basename(filenames[i]))
            elif y_pred[i] == 0 and y_true[i] == 1:
                dst = os.path.join(output_dir_fn, os.path.basename(filenames[i]))
            shutil.copyfile(src, dst)
            
    
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return accuracy, auc, fp, fn, tn ,tp


  
# Evaluate the pre-trained model
accuracy_pre, auc_pre, fp_pre, fn_pre, tn_pre, tp_pre = evaluate_model(model_pre, test_gen, "../results/pre")
print(f'Pre-trained Model - Accuracy: {accuracy_pre:.4f}, AUC Score: {auc_pre:.4f}, False Positives: {fp_pre}, False Negatives: {fn_pre}, True Postives: {tp_pre}, True Negatives: {tn_pre}')

# Evaluate the fully trained model
accuracy_post, auc_post, fp_post, fn_post, tn_post, tp_post  = evaluate_model(model_post, test_gen,  "../results/post")
print(f'Fully Trained Model - Accuracy: {accuracy_post:.4f}, AUC Score: {auc_post:.4f}, False Positives: {fp_post}, False Negatives: {fn_post}, True Postives: {tp_post}, True Negatives: {tn_post}')


print(test_gen.classes, sum(test_gen.classes), len(test_gen.classes) - sum(test_gen.classes))