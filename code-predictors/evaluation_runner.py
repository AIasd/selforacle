import logging

import numpy

import training_runner
import utils_args
import utils_logging
from eval_db.database import Database
from eval_db.eval_seq_img_distances import SeqBasedDistance
from eval_db.eval_setting import Setting
from eval_db.eval_single_img_distances import SingleImgDistance

# addition
import argparse
import numpy as np
import random
import os

# modification
parser = argparse.ArgumentParser(description='Behavioral Cloning Evaluating Program')
# should let this equal to labeling when not running baseline
parser.add_argument('-m', help='mode for evaluation', dest='mode', type=str, default='')
args = parser.parse_args()
mode = args.mode

if mode == 'labeling':
    train_args = utils_args.specify_args()
else:
    train_args = utils_args.load_train_args()

train_args.always_calc_thresh = False
train_args.delete_trained = False





logger = logging.Logger("main")
utils_logging.log_info(logger)
# modification: ['SAE', "VAE", 'CAE', "DAE", "DEEPROAD"] -> ['SAE']
SINGLE_IMAGE_ADS = ['SAE']
# modification: ["IMG-LSTM"] -> []
SEQUENCE_BASED_ADS = []


EVAL_AGENTS = ["LBC"]

eval_dir = "../../2020_CARLA_challenge/collected_data_customized/customized_0"
EVAL_TRACKS = []
for route in sorted(os.listdir(eval_dir)):
    route_path = eval_dir+'/'+route
    if os.path.isdir(route_path):
        EVAL_TRACKS.append(route)

SETTING_START_ID = 3000



def main():
    train_data_dir = train_args.data_dir
    # train_dataset_name = training_runner.dataset_name_from_dir(train_data_dir)
    db_name = "../models/trained-anomaly-detectors/collected_data_customized-based-eval.sqlite"

    # Prepare Database
    db = Database(db_name, True)
    # Prepare Settings
    settings = _create_all_settings(db)

    # Prepare ADs

    single_img_based_ads, sequence_based_ads = _prepare_ads(train_data_dir, train_args)

    for i, setting in enumerate(settings):
        data_dir = os.path.join(eval_dir, setting.get_folder_name())

        if len(single_img_based_ads) > 0:
            handle_single_image_based_ads(db=db, data_dir=data_dir, setting=setting, single_img_based_ads=single_img_based_ads, raw_data_dir=eval_dir, mode=mode)

        if len(sequence_based_ads) > 0:
            handle_sequence_based_ads(db=db, data_dir=data_dir, setting=setting, sequence_based_ads=sequence_based_ads, raw_data_dir=eval_dir)

    # print('single_img_entries')
    # get_current_single_img_entries_num(db, 3000)
    # get_current_single_img_entries_num(db, 3001)
    # get_current_single_img_entries_num(db, 3002)


def handle_sequence_based_ads(db, data_dir, setting, sequence_based_ads, raw_data_dir):
    ad_distances = {}
    frame_ids = None
    are_crashes = None
    setting_name = setting.get_folder_name()

    for ad_name, ad in sequence_based_ads.items():
        logger.info("Calculating losses for " + setting.get_folder_name() + " with ad  " + ad_name)
        x, y, frm_ids, crashes = ad.load_img_paths(data_dir=data_dir, restrict_size=False, eval_data_mode=True)
        assert len(x) == len(y) == len(frm_ids) == len(crashes)
        case_name = setting_name+'/'+ad_name
        distances = ad.calc_losses(inputs=x, labels=y, data_dir=raw_data_dir, case_name=case_name)
        ad_distances[ad_name] = distances
        if frame_ids is None:
            frame_ids = frm_ids
            are_crashes = crashes
    logger.info("Done. Now storing sequence based eval for setting " + setting.get_folder_name())
    store_seq_losses(setting=setting, per_ad_distances=ad_distances, row_ids=frame_ids, are_crashes=are_crashes, db=db)


def handle_single_image_based_ads(db, data_dir, setting, single_img_based_ads, raw_data_dir, mode=None):
    ad_distances = {}
    frame_ids = None
    are_crashes = None
    setting_name = setting.get_folder_name()

    for ad_name, ad in single_img_based_ads.items():
        logger.info("Calculating losses for " + setting_name + " with ad  " + ad_name)

        x, frm_ids, crashes = ad.load_img_paths(data_dir=data_dir, restrict_size=False, eval_data_mode=True)


        assert len(x) == len(frm_ids) == len(crashes)
        if mode == 'labeling':
            distances = np.zeros_like(x)
        else:
            case_name = setting_name+'/'+ad_name
            distances = ad.calc_losses(inputs=x, labels=None, data_dir=raw_data_dir, case_name=case_name)


        ad_distances[ad_name] = distances
        if frame_ids is None:
            frame_ids = frm_ids
            are_crashes = crashes
    logger.info("Done. Now storing single img based eval for setting " + setting_name)
    store_losses(setting=setting, per_ad_distances=ad_distances, row_ids=frame_ids, are_crashes=are_crashes, db=db)


def store_seq_losses(setting, per_ad_distances, row_ids, are_crashes, db: Database):
    for i in range(len(per_ad_distances["IMG-LSTM"])):
        setting_id = setting.id
        row_id = row_ids[i]
        row_id = row_id.item()
        if are_crashes[i] == 0:
            is_crash = False
        else:
            is_crash = True
        lstm_loss = per_ad_distances["IMG-LSTM"][i]
        to_store = SeqBasedDistance(setting_id=setting_id, row_id=row_id, is_crash=is_crash, lstm_loss=lstm_loss)
        to_store.insert_into_db(db)
        if i % 1000:
            db.commit()
    db.commit()


def store_losses(setting, per_ad_distances, row_ids, are_crashes, db: Database):
    assert per_ad_distances
    # modification: len(per_ad_distances["VAE"]) -> len(list(per_ad_distances.items())[0][1])
    for i in range(len(list(per_ad_distances.items())[0][1])):
        setting_id = setting.id
        row_id = row_ids[i]
        row_id = row_id.item()
        if are_crashes[i] == 0:
            is_crash = False
        else:
            is_crash = True
        # add if conditions to make these losses are None when they are not evaluated
        vae_loss = None
        sae_loss = None
        cae_loss = None
        dae_loss = None
        deeproad_loss = None
        if "VAE" in per_ad_distances:
            vae_loss = per_ad_distances["VAE"][i]
        if "SAE" in per_ad_distances:
            sae_loss = per_ad_distances["SAE"][i]
        if "CAE" in per_ad_distances:
            cae_loss = per_ad_distances["CAE"][i]
        if "DAE" in per_ad_distances:
            dae_loss = per_ad_distances["DAE"][i]
        if "DEEPROAD" in per_ad_distances:
            deeproad_loss = per_ad_distances["DEEPROAD"][i]
            deeproad_loss = deeproad_loss.item()
        to_store = SingleImgDistance(setting_id=setting_id, row_id=row_id, is_crash=is_crash, vae_loss=vae_loss, cae_loss=cae_loss, dae_loss=dae_loss, sae_loss=sae_loss, deeproad_loss=deeproad_loss)
        to_store.insert_into_db(db)
        if i % 1000:
            db.commit()
    db.commit()



def _prepare_ads(data_dir, train_args):
    single_img_ads = {}
    for ad_name in SINGLE_IMAGE_ADS:
        single_img_ads[ad_name] = training_runner.load_or_train_model(args=train_args, data_dir=data_dir, model_name=ad_name)
    sequence_based_ads = {}
    logger.warning("Enable squence based again")
    for ad_name in SEQUENCE_BASED_ADS:
        sequence_based_ads[ad_name] = training_runner.load_or_train_model(args=train_args, data_dir=data_dir, model_name=ad_name)
    return single_img_ads, sequence_based_ads


def _create_all_settings(db: Database):
    settings = []
    id = SETTING_START_ID
    for agent in EVAL_AGENTS:
        for track in EVAL_TRACKS:
            setting = Setting(id=id, agent=agent, track=track)
            setting.insert_into_db(db=db)
            id = id + 1
            settings.append(setting)
    db.commit()
    return settings

# addition:
def get_current_single_img_entries_num(db, setting_id):
    cursor = db.cursor.execute('select * from single_image_based_distances where setting_id=? ' +
                               'order by row_id',
                               (setting_id,))
    var = cursor.fetchall()
    result = []
    # addition
    print('setting id :', setting_id)
    print('var :', len(var))

if __name__ == '__main__':
    main()
