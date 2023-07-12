import argparse


def load_pickle(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)


class TaskConfig:
    num_class = None
    train_file = None
    val_file = None
    test_file = None
    sequence_len = None
    eval_interval = None
    patience = None
    balance = None


def get_task_config(task_name):
    config = TaskConfig()
    if task_name == 'atk-pattern':
        config.num_class = 2
        config.train_file = 'data/atk-pattern/train.pickle'
        config.val_file = 'data/atk-pattern/validation.pickle'
        config.test_file = 'data/atk-pattern/test.pickle'
        config.sequence_len = 256
        config.eval_interval = 10
        config.patience = 10
        config.balance = False
    else:
        raise ValueError('Task not supported')
    return config
