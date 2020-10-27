'''
* causal evaluation pre-implementing
* fix to support run multiple route consecutively
* read RL paper
* read Rep class paper


* reformat customized data and run action based model; apply the trained encoder to train another model for predicting misbehaviors
* clean up data / code and share with Conor; ask him to 1.analyze correlation between variables (control, speed) and normal / anormaly frames 2.ask him to try to rewrite the labeling normal / anormaly script and then rerun the baseline in a non-database fashion? 3.potentially ask him to play with the model and data, insight on data and give feedback


* add contrastive loss into the training


* use customized data for training encoder for action based (probably need to understand their pipeline code first).

* collect data using expert driver and remove those redundant data / misbehavior data, then train another encoder.

* (record affordance info and try to predict them).
* add more heads and check performance change.

* integrate pipeline with misbehavior prediction and check performance.




* remove those frames where the car barely moves
* build encoder that predict control
* make prediction scenario even simpler and concatenate more information (speed, control) to the simple CNN classifier

* reimplement the labeling procedure? reimplement the baseline measure since labeling is reimplemented without relying on the sqlite database?
* get simple_detector work (try limiting type of crash as well as removing redundant frames)
* try baseline on the updated data
* try to somehow unify the pipeline
* update readme for easier use
* fix simulator crush error after 30-50 simulations
* support adding background ped
* avoid generating other objects too close to the ego car for causal inference case; let the other car to follow traffic rules when waypoint follower is invoked; measure ego car width and length


* process data (train/test split, complete data, remove redundant repetitions) to make performance non-trivial
* simple classifier on it
* try the proposed model
* try more fine-grained output

* detection use all three front cameras
'''

import logging
import os
import random
import numpy as np

import utils
import utils_args
from detectors.anomaly_detector import AnomalyDetector
from detectors.img_sequence_cnnlstm.cnn_lstm_img import CnnLstmImg
from detectors.single_image_based_detectors.autoencoders.convolutional_autoencoder import ConvolutionalAutoencoder
from detectors.single_image_based_detectors.autoencoders.deep_autoencoder import DeepAutoencoder
from detectors.single_image_based_detectors.autoencoders.simple_autoencoder import SimpleAutoencoder
from detectors.single_image_based_detectors.autoencoders.variational_autoencoder import VariationalAutoencoder
from detectors.single_image_based_detectors.deeproad_rebuild.deeproad import Deeproad

logger = logging.Logger("main")


def get_model(args, model_name, data_dir):
    dataset_name = dataset_name_from_dir(data_dir)
    if model_name == 'CAE':
        return ConvolutionalAutoencoder(name="convolutional-autoencoder-model-" + dataset_name, args=args)
    elif model_name == 'SAE':
        return SimpleAutoencoder(name="simple-autoencoder-model-" + dataset_name, args=args)
    elif model_name == "VAE":
        return VariationalAutoencoder(name="variational-autoencoder-model-" + dataset_name, args=args)
    elif model_name == "DAE":
        return DeepAutoencoder(name="deep-autoencoder-model-" + dataset_name,
                               args=args, hidden_layer_dim=256)
    elif model_name == "IMG-LSTM":
        return CnnLstmImg(name="LSTM-model-" + dataset_name, args=args)

    elif model_name == "DEEPROAD":
        return Deeproad(name="deeproad-pca-model-" + dataset_name, args=args)

    else:
        logger.error("Unknown Model Type: " + model_name)


def dataset_name_from_dir(data_dir):
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    return dataset_name


def main():
    np.random.seed(0)
    args = utils_args.specify_args()
    utils_args.store_and_print_params(args)


    for model_name in args.model_name:
        print("\n --- MODEL NAME: " + model_name + " for dataset " + data_dir + " --- \n")
        # Load the correct anomaly detector class
        load_or_train_model(args, args.data_dir, model_name)
        print("\n --- COMPLETED  " + model_name + " for dataset " + data_dir + " --- \n")

    print("\ndone")
    # evaluate_optically_img_loss(trained=autoencoder, x_test=X_test, y_test=y_test, args=args)


def load_or_train_model(args, data_dir, model_name) -> AnomalyDetector:
    anomaly_detector = get_model(args=args, model_name=model_name, data_dir=data_dir)
    # Create and compile the ADs model
    anomaly_detector.initialize()
    # Load the image paths in a form suitable for the AD (sequence or single-image)
    #       Indicate whether this models requires special TD size
    restrict_size = args.train_abs_size_models.count(model_name) > 0


    # Load previously trained model or train it now
    anomaly_detector.load_or_train_model(restrict_size=restrict_size, data_dir=data_dir)

    # Sanity check for loss calculator
    # anomaly_detector.calc_losses(x_train[:200], y_train[:200], data_dir=data_dir)
    return anomaly_detector

if __name__ == '__main__':
    main()
