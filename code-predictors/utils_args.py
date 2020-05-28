import argparse
import logging
import os
import pickle

import utils_logging




MODELS = ['SAE', "VAE", 'CAE', "DAE", "DEEPROAD", "IMG-LSTM"]
SIMPLE_MODELS_ONLY = ['SAE', "VAE", 'CAE', "DAE"]

logger = logging.Logger("Args_utils")
utils_logging.log_info(logger)


# addition
def get_args_serialization_path(simulator):
    # TBD: changed to not-hardcoded
    if simulator == 'udacity':
        return "/home/zhongzzy9/Documents/self-driving-car/misbehavior_prediction//models/trained-anomaly-detectors/udacity_training-args.pkl"
    elif simulator == 'carla_096':
        return "/home/zhongzzy9/Documents/self-driving-car/misbehavior_prediction//models/trained-anomaly-detectors/carla_096_training-args.pkl"
    elif simulator == 'carla_099':
        return "/home/zhongzzy9/Documents/self-driving-car/misbehavior_prediction//models/trained-anomaly-detectors/carla_099_training-args.pkl"

def store_and_print_params(args):
    _print_parameters(args)
    # Check consistency with stored args
    path = get_args_serialization_path(args.simulator)
    if os.path.exists(path):
        stored_args = load_train_args(args.simulator)
        if not args == stored_args:
            logger.error(
                "The stored settings " + path
                + " do not match the specified args. "
                + "If this is intended, please delete the stored args. "
                + "We recommend also deleting the previously trained models.")
    else:
        _write_train_args(args=args)


def _print_parameters(args):
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)


def specify_args():
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', nargs='+', help='data directory', dest='data_dir', type=str,
                        default=['../datasets/dataset5'])
    # modification
    parser.add_argument('-trs', help='restrict train set size, -1 if none', dest='train_abs_size', type=int,
                        default=10000)
    parser.add_argument('-trm', nargs='+', help='restrict train set size for models', dest='train_abs_size_models',
                        # type=str, default=MODELS) #modification
                        type=str, default=["SAE"])

    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=2)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=32)
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-m', nargs='+', help='model name', dest='model_name', type=str,
                        # default=MODELS) #modification
                        default=['SAE'])
    parser.add_argument('-r', help='random state', dest='random_state', type=int, default=0)
    parser.add_argument('-t', help="force recalc of thresholds on model relaod", dest="always_calc_thresh", type=s2b, default=True)
    parser.add_argument('-sl', help='sequence length', dest='sequence_length', type=int, default=30)
    parser.add_argument('-dl', help='delete trained model', dest='delete_trained', type=s2b, default='true')
    parser.add_argument('-g', help='gray scale image', dest='gray_scale', type=s2b, default='false')
    # addition
    parser.add_argument('-sim', help='simulator used to generate data', dest='simulator', type=str, default='udacity')
    args = parser.parse_args()
    return args

def s2b(s):
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def load_train_args(simulator):
    path = get_args_serialization_path(simulator)
    with open(path, 'rb') as input:
        if os.path.getsize(path) > 0:
            return pickle.load(input)
        else:
            return None  #Persisted file was empty


def _write_train_args(args) -> None:
    path = get_args_serialization_path(args.simulator)
    with open(path, 'wb+') as output:
        pickle.dump(args, output, pickle.HIGHEST_PROTOCOL)
