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
from recbole.utils import init_seed, init_logger, get_model
from recbole.utils.case_study import full_sort_scores, full_sort_topk
from recbole.quick_start import load_data_and_model
from recbole.data.interaction import Interaction
from utils import *

model_type_list = ['sequential', 'general', 'context_aware', 'knowledge_aware', 'exlib']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='boostcamp', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    parser.add_argument('--type', type=str, default='G', help='S: Sequential, G: General, C: Context-aware, K:Knowledge-aware, E: exlib')
    args = parser.parse_args()
    
    if args.config_files.endswith('.yaml'):
        args.config_files = os.path.join('./config', args.config_files)
    else:
        args.config_files = os.path.join('./config', args.config_files + ".yaml")

    BASE_DIR = "./saved"
    FILE = ""
    
    file_list = os.listdir(BASE_DIR)

    if len(file_list) > 1:
        assignment = True
        while assignment:
            print('Select "number" you want to do')
            for i in range(len(file_list)):
                print(f'[{i}] : {file_list[i]}')

            number = input()
            try:
                number = int(number)
                FILE = file_list[number]
                assignment= False
            except:
                pass
    elif len(file_list) == 0:
        sys.exit('There is no .pth file.')
    else:
        FILE = file_list[0]
    
    model_name = FILE.split("-")[0]
    model_type = ''

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=os.path.join(BASE_DIR, FILE)
    )
    
    imported_model = get_model(model_name)
    process = str(imported_model).split('.')

    for name in process:
        if 'recommender' in name:
            model_type = name
            break
    else:
        print('Model name is not found.')



    config = Config(model=model_name, dataset=args.dataset, config_file_list=[args.config_files])
    dataset = create_dataset(config)

    train_data, valid_data, test_data = data_preparation(config, dataset)


    if model_type in model_type_list[0]:
        prediction = generate_predict_seq(dataset, test_data, model, config)
    else:
        prediction = generate_predict(dataset, test_data, model, config)

    gererate_submission_from_prediction(prediction=prediction)
