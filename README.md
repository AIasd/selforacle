# Misbehaviour Prediction for Autonomous Driving Systems

This repository contains the code modified from `Misbehaviour Prediction for Autonomous Driving Systems`.

## Setup
Install packages as specified in `environment.yml`.

## Using new data (carla 0.9.9)
###Preprocessing collected data
In 2020_CARLA_challenge repo, run
```
python process_collected_data.py
```

setting training data path in `code-predictors/detectors/single_image_based_detectors/abs_single_image_autoencoder.py`.

setting testing data path in `code-predictors/evaluation_runner.py`.
### Training an anomaly detector
In the selforacle repo, run
```
python code-predictors/training_runner.py -d '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/collected_data_customized' -sim carla_099 -trs=-1
```
### Testing an anomaly detector
In the selforacle repo, run

```
python code-predictors/evaluation_runner.py
python code-predictors/eval_scripts/a_set_true_labels.py
python code-predictors/eval_scripts/b_precision_recall_auroc.py
```

## Pre-processing data for our detection method
In the repo 2020_CARLA_challenge
```
python process_collected_data.py
```

Customize path in the function `get_args_serialization_path` of `utils_args.py`

In `code-predictors/evaluation_runner.py`, set data path for `eval_dir`

In `code-predictors/a_set_true_labels.py`, set data path for `driving_log`

In the repo selforacle
```
python code-predictors/evaluation_runner.py -m labeling
python code-predictors/eval_scripts/a_set_true_labels.py
```

### Train and Test Supervised Detector
In `simple_detector.py`, set path for `total_data_dir`, `train_data_dir`, and `test_data_dir`

```
python code-predictors/customized_detectors/simple_detector.py
```



<!-- ### Extract feature vectors
After the previous steps of "Training an anomaly detector" and "Pre-processing data for our detection method", in the selforacle repo, run
```
python code-predictors/evaluation_runner.py -sim carla_099
```
In the repo 2020_CARLA_challenge, run
```
python process_feature_vectors.py
``` -->



## Reference
The current repo is forked from the repo for the following paper:

```
@inproceedings{2020-icse-misbehaviour-prediction,
	title= {Misbehaviour Prediction for Autonomous Driving Systems},
	author= {Andrea Stocco and Michael Weiss and Marco Calzana and Paolo Tonella},
	booktitle= {Proceedings of 42nd International Conference on Software Engineering},
	series= {ICSE '20},
	publisher= {ACM},
	pages= {12 pages},
	year= {2020}
}
```
