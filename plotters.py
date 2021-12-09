import argparse
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


def plot_acc_metric(history):
    plt.plot(history['accuracy'], label="Training")
    plt.plot(history['val_accuracy'], label="Validation")
    plt.legend()
    plt.title("Model accuracy over Epochs")
    plt.ylabel("Accuracy (Normalized)")
    plt.xlabel("Epoch")
    plt.xticks(range(0, len(history['accuracy'])))
    plt.show()


def plot_loss_metric(history):
    plt.plot(history['loss'], label="Training")
    plt.plot(history['val_loss'], label="Validation")
    plt.title("Model loss over Epochs")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.xticks(range(0, len(history['loss'])))
    plt.legend()
    plt.show()



def script_invocation():
    parser = argparse.ArgumentParser(description="Plot metrics from model history")

    parser.add_argument("-m", "--model_path", help="path of model containing the history", type=str, default=None)
    parser.add_argument("-a", "--plot_acc", help="Plot the model accuracy over epochs", action="store_true")
    parser.add_argument("-l", "--plot_loss", help="Plot the loss over epochs", action="store_true")

    args = parser.parse_args()

    if args.model_path is None:
        return

    hist = pd.read_csv(args.model_path, delimiter=",")
    print(hist)

    if args.plot_acc:
        plot_acc_metric(hist)

    if args.plot_loss:
        plot_loss_metric(hist)


if __name__ == '__main__':
    script_invocation()