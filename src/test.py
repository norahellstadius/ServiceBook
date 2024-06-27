import os 
import shutil
import numpy as np 
from typing import Tuple
from utils import check_dir

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


def datagen(test_dir: str, batchSize: int =32, pixel: int =224):
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
    test_gen = datagenerator.flow_from_directory(test_dir, target_size=(pixel, pixel), class_mode='binary', batch_size=batchSize, shuffle=False)
    return test_gen

def evaluate_model(dir_test: str, model, test_gen: DirectoryIterator, output_dir: str) -> Tuple[float, float, int, int, int, int]:
    """
    Evaluate a model by predicting on a test dataset, calculating metrics, and saving misclassified images.

    Args:
        dir_test (str): Directory containing the test images.
        model: Trained model to evaluate.
        test_gen (DirectoryIterator): Test data generator.
        output_dir (str): Directory to save the evaluation results.

    Returns:
        Tuple[float, float, int, int, int, int]: Accuracy, AUC, false positives, false negatives, true negatives, true positives.
    """
    check_dir(output_dir)

    y_pred_prob = model.predict(test_gen)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = test_gen.classes
    filenames = test_gen.filenames

    output_dir_fp = os.path.join(output_dir, "false_positives")
    output_dir_fn = os.path.join(output_dir, "false_negatives")
    check_dir(output_dir_fp)
    check_dir(output_dir_fn)

    # Identify misclassified images and copy them
    for i in range(len(y_true)):
        if y_pred[i] != y_true[i]:
            src = os.path.join(dir_test, filenames[i])
            if y_pred[i] == 1 and y_true[i] == 0:
                dst = os.path.join(output_dir_fp, os.path.basename(filenames[i]))
                shutil.copyfile(src, dst)
                print(f"Copied false positive: {filenames[i]} to {output_dir_fp}")
            elif y_pred[i] == 0 and y_true[i] == 1:
                dst = os.path.join(output_dir_fn, os.path.basename(filenames[i]))
                shutil.copyfile(src, dst)
                print(f"Copied false negative: {filenames[i]} to {output_dir_fn}")

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
    plt.yticks(tick_marks, ['Negative', 'Positive'])

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    # Save confusion matrix plot
    cm_image_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_image_path)
    plt.close()
    print(f"Confusion matrix saved to: {cm_image_path}")

    return accuracy, auc, fp, fn, tn, tp


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument("--test_dir", type=str, default="../data/split/test", help="Directory to the test set")
    parser.add_argument("--pixel", type=int, default=224, help="Pixel dimensions for the input images")
    parser.add_argument("--batch_size", type=int, default=32, help="Pixel dimensions for the input images")


    args = parser.parse_args()

    # Generate the test set
    test_gen = datagen(args.test_dir, batchSize=args.batch_size, pixel=args.pixel)

    # Load the pre-trained model
    model_pre = tf.keras.models.load_model('../models/modelResNet_50_pre.keras')
    # Load the fully trained model
    model_post = tf.keras.models.load_model('../models/modelResNet_50_post.keras')

    # Evaluate the pre-trained model
    accuracy_pre, auc_pre, fp_pre, fn_pre, tn_pre, tp_pre = evaluate_model(args.test_dir, model_pre, test_gen,  "../results/pre")
    print(f'Pre-trained Model - Accuracy: {accuracy_pre:.4f}, AUC Score: {auc_pre:.4f}, False Positives: {fp_pre}, False Negatives: {fn_pre}, True Positives: {tp_pre}, True Negatives: {tn_pre}')

    # Evaluate the fully trained model
    accuracy_post, auc_post, fp_post, fn_post, tn_post, tp_post = evaluate_model(args.test_dir, model_post, test_gen, "../results/post")
    print(f'Fully Trained Model - Accuracy: {accuracy_post:.4f}, AUC Score: {auc_post:.4f}, False Positives: {fp_post}, False Negatives: {fn_post}, True Positives: {tp_post}, True Negatives: {tn_post}')

