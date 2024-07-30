import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve, precision_recall_curve


# confusion matrix code from Maurizio
# /eos/user/m/mpierini/DeepLearning/ML4FPGA/jupyter/HbbTagger_Conv1D.ipynb
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    cbar = plt.colorbar()
    plt.clim(0, 1)
    cbar.set_label(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plotRoc(fpr, tpr, auc, labels, linestyle, legend=True):
    for _i, label in enumerate(labels):
        plt.plot(
            tpr[label],
            fpr[label],
            label='{} tagger, AUC = {:.1f}%'.format(label.replace('j_', ''), auc[label] * 100.0),
            linestyle=linestyle,
        )
    plt.semilogy()
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Efficiency")
    plt.ylim(0.001, 1)
    plt.grid(True)
    if legend:
        plt.legend(loc='upper left')
    plt.figtext(0.25, 0.90, 'hls4ml', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)


def rocData(y, predict_test, labels):
    df = pd.DataFrame()

    fpr = {}
    tpr = {}
    auc1 = {}

    for i, label in enumerate(labels):
        df[label] = y[:, i]
        df[label + '_pred'] = predict_test[:, i]

        fpr[label], tpr[label], threshold = roc_curve(df[label], df[label + '_pred'])

        auc1[label] = auc(fpr[label], tpr[label])
    return fpr, tpr, auc1


def makeRoc(y, predict_test, labels, linestyle='-', legend=True):
    if 'j_index' in labels:
        labels.remove('j_index')

    fpr, tpr, auc1 = rocData(y, predict_test, labels)
    plotRoc(fpr, tpr, auc1, labels, linestyle, legend=legend)
    return predict_test


def print_dict(d, indent=0):
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))
            
            
def roc_fig_plot(y_true_l, **kwargs):
    """
    Plot ROC curves for multiple classifiers.
    
    Parameters:
    y_true_l (list): List of true labels.
    kwargs (dict): Dictionary of classifier names and their scores.
    
    Returns:
    None
    """
    plt.rcParams["figure.figsize"] = (10, 10)
    styles = ["-", "--"]
    colors = ["b", "g", "r", "c", "m", "y", "k", "orange"]

    for i, (key, value) in enumerate(kwargs.items()):
        assert len(y_true_l[i]) == len(value), "Length of true labels and scores must be equal"

        style = styles[i % 2]
        color_idx = math.floor(i / 2)
        color = colors[color_idx % len(colors)]

        fpr, tpr, thresholds = roc_curve(y_true_l[i], value)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, style, label=f'AUC {key.replace("_", " ")} = {round(roc_auc, 3)}', color=color)

    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate (TPR)')
    plt.xlabel('False Positive Rate (FPR)')
    plt.show()
    
def precision_recall_fig_plot(y_true_l, **kwargs):
    """
    Plot Precision-Recall curves for multiple classifiers.
    
    Parameters:
    y_true_l (list): List of true labels.
    kwargs (dict): Dictionary of classifier names and their scores.
    
    Returns:
    None
    """
    plt.rcParams["figure.figsize"] = (10, 10)
    styles = ["-", "--"]
    colors = ["b", "g", "r", "c", "m", "y", "k", "orange"]

    for i, (key, value) in enumerate(kwargs.items()):
        assert len(y_true_l[i]) == len(value), "Length of true labels and scores must be equal"

        style = styles[i % 2]
        color_idx = math.floor(i / 2)
        color = colors[color_idx % len(colors)]

        precision, recall, thresholds = precision_recall_curve(y_true_l[i], value)
        prc_auc = auc(recall, precision)
        plt.plot(recall, precision, style, label=f'AUC {key.replace("_", " ")} = {round(prc_auc, 3)}', color=color)

    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()


def plot_training_history(history):
    plt.figure(figsize=(12, 8))

    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history.get('val_loss'), label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot additional metrics
    plt.subplot(2, 1, 2)
    plt.plot(history['mse'], label='Training MSE')
    plt.plot(history.get('val_mse'), label='Validation MSE')
    plt.plot(history['mae'], label='Training MAE')
    plt.plot(history.get('val_mae'), label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Training and Validation Metrics')
    plt.legend()

    plt.tight_layout()
    plt.show()