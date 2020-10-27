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
def get_args_serialization_path():
    # TBD: changed to be not-hardcoded
    return "../models/trained-anomaly-detectors/carla_099_training-args.pkl"

def store_and_print_params(args):
    _print_parameters(args)
    path = get_args_serialization_path()
    _write_train_args(args=args)


def _print_parameters(args):
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'

def specify_args():
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str,
                        default='/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/collected_data_customized')
    # modification
    parser.add_argument('-trs', help='restrict train set size, -1 if none', dest='train_abs_size', type=int,
                        default=1000)
    parser.add_argument('-trm', nargs='+', help='restrict train set size for models', dest='train_abs_size_models', type=str, default=MODELS)

    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=2)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=32)
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-m', nargs='+', help='model name', dest='model_name', type=str,
                        # default=MODELS) #modification
                        default=['SAE'])
    # parser.add_argument('-r', help='random state', dest='random_state', type=int, default=0)
    parser.add_argument('-t', help="force recalc of thresholds on model reload", dest="always_calc_thresh", type=s2b, default=True)
    parser.add_argument('-sl', help='sequence length', dest='sequence_length', type=int, default=30)
    parser.add_argument('-dl', help='delete trained model', dest='delete_trained', type=s2b, default='true')
    # parser.add_argument('-g', help='gray scale image', dest='gray_scale', type=s2b, default='false')
    # addition
    # parser.add_argument('-sim', help='simulator used to generate data', dest='simulator', type=str, default='udacity')
    # parser.add_argument('-weather', nargs='+', help='weather_indexes', dest='weather_indexes', type=int, default=[15])
    # parser.add_argument('-route', nargs='+', help='route_indexes', dest='route_indexes', type=int, default=[i for i in range(30)])
    args = parser.parse_args()
    return args


def load_train_args():
    path = get_args_serialization_path()
    with open(path, 'rb') as input:
        if os.path.getsize(path) > 0:
            return pickle.load(input)
        else:
            return None  #Persisted file was empty


def _write_train_args(args) -> None:
    path = get_args_serialization_path()
    with open(path, 'wb+') as output:
        pickle.dump(args, output, pickle.HIGHEST_PROTOCOL)
