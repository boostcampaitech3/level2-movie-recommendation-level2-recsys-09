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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='boostcamp', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    if args.config_files.endswith('.yaml'):
        args.config_files = os.path.join('./config', args.config_files)
    else:
        args.config_files = os.path.join('./config', args.config_files + ".yaml")

    BASE_DIR = "./saved"
    FILE = "LightSANs-Mar-29-2022_15-07-16.pth"
    model_name = FILE.split("-")[0]

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=os.path.join(BASE_DIR, FILE)
    )

    config = Config(model=model_name, dataset=args.dataset, config_file_list=[args.config_files])
    dataset = create_dataset(config)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    prediction = generate_predict(dataset, test_data, model_name, config)
    gererate_submission_from_prediction(prediction=prediction)
