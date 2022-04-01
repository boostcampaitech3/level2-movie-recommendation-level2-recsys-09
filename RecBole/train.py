import argparse
from recbole.config import Config
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.model.sequential_recommender import GRU4Rec, LightSANs
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parameter_dict = {
        "data_path": "./data/",
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "user_inter_num_interval": "[0,inf)",
        "item_inter_num_interval": "[0,inf)",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "neg_sampling": {"uniform": 1},
        "loss_type": "BPR",
        "epochs": 50,
        "train_batch_size": 256,
        "eval_batch_size": 4096,
        "MAX_ITEM_LIST_LENGTH": 50,
        "k_interests": 5, # for LightSANs (MAX_ITEM_LIST_LENGTH * 0.1)
        "eval_args": {
            "split": {"RS": [9, 1, 0]},
            "group_by": "user",
            "order": "TO",
            "mode": "full",
        }
    }

    config = Config(model="LightSANs", dataset="boostcamp", config_dict=parameter_dict)

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
    model = LightSANs(config, train_data.dataset).to(config["device"])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    # test_result = trainer.evaluate(test_data)
    # print(test_result)
