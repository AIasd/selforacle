# modification:
# udacity
DB_PATH = "../models/trained-anomaly-detectors/udacity-dataset5-based-eval.sqlite"
# carla_096
# DB_PATH = "../models/trained-anomaly-detectors/collected_data-based-eval.sqlite"



def get_db_path(simulator):
    if simulator == 'udacity':
        return "../models/trained-anomaly-detectors/udacity-dataset5-based-eval.sqlite"
    elif simulator == 'carla_096':
        return "../models/trained-anomaly-detectors/carla_096-collected_data-based-eval.sqlite"
    elif simulator == 'carla_099':
        return "../models/trained-anomaly-detectors/carla_099-collected_data-based-eval.sqlite"
    else:
        raise
