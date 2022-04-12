import argparse
import os
from recbole.config import Config
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer, RecVAETrainer, PretrainTrainer, KGATTrainer, RaCTTrainer
from recbole.utils import init_seed, init_logger, get_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BPR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='boostcamp', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    parser.add_argument('--lambda2', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--rho', type=int, default=None)
    
    args = parser.parse_args()
    
    if args.config_files.endswith('.yaml'):
        args.config_files = os.path.join('./config', args.config_files)
    else:
        args.config_files = os.path.join('./config', args.config_files + ".yaml")
    
    config = Config(model=args.model, dataset=args.dataset, config_file_list=[args.config_files])
    
    config['lambda2'] = args.lambda2
    config['alpha'] = args.alpha
    config['rho'] = args.rho
    
    # init random seed
    init_seed(config["seed"], config["reproducibility"])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    imported_model = get_model(args.model)
    model = imported_model(config, train_data.dataset).to(config["device"])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    
    f = open("save.txt", 'a')
    note = "lambda2 : " + str(args.lambda2) + " alpha : " + str(args.alpha) + " rho : " + str(args.rho) + " score : " + str(best_valid_score) 
    f.write('\n' + note + '\n')
    f.close()
    # model evaluation
    # test_result = trainer.evaluate(test_data)
    # print(test_result)
