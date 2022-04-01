import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from recbole.config import Config
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.utils.case_study import full_sort_scores, full_sort_topk
from recbole.quick_start import load_data_and_model
from recbole.data.interaction import Interaction
from utils import *


if __name__ == "__main__":

    BASE_DIR = "/opt/ml/input/data/recbole/saved"
    FILE = "LightSANs-Mar-29-2022_15-07-16.pth"

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=os.path.join(BASE_DIR, FILE)
    )

    parameter_dict = {
        "data_path": "./data/",
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "user_inter_num_interval": "[0,inf)",
        "item_inter_num_interval": "[0,inf)",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "neg_sampling": None,
        "epochs": 5,
        "MAX_ITEM_LIST_LENGTH": 50,
        "eval_args": {
            "split": {"RS": [0, 0, 1]},
            "group_by": "user",
            "order": "TO",
            "mode": "full",
        },
    }

    config = Config(model="LightSANs", dataset="boostcamp", config_dict=parameter_dict)
    dataset = create_dataset(config)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    prediction = generate_predict(dataset, test_data, model, config)
    gererate_submission_from_prediction(prediction=prediction)
