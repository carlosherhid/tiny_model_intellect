import click
import os
import random
import shutil
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer_names
from autogluon.tabular import TabularPredictor
from utils.ranking import get_all_ranking_algorithms, get_ranking_algorithm
from utils.model import test_model_on_subset

def create_dir(directory, overwrite=False):
    if not os.path.exists(directory):
        os.makedirs(directory)
    elif overwrite:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        os.makedirs(directory)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def get_default_cpu_number():
    return os.cpu_count()

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")

def draw_best(scores, top=True):
    scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    comparison_value = scores_sorted[0][1] if top else scores_sorted[-1][1]
    return random.choice([x[0] for x in scores_sorted if x[1] == comparison_value])

def rank(predictor_directory: str, dataset_directory: str, algorithm: str, metric: str, cpus: int, recursive: bool, time_limit: int):
    global logger
    set_seed()

    logger.info('Loading Model')
    predictor = TabularPredictor.load(predictor_directory)
    best_model_name = predictor._trainer.model_best
    best_model = predictor._trainer.load_model(best_model_name)

    logger.info('Loading data')
    X = pd.read_csv(os.path.join(dataset_directory, 'finetune.csv'), index_col='ID')
    y = X.pop('Label')

    store_path = os.path.join(predictor_directory, f'feature_ranking_{algorithm}')
    create_dir(store_path, overwrite=False)

    is_sbe = algorithm.endswith('sbe')
    algorithm_func = get_ranking_algorithm(algorithm)

    current_features = X.columns.values.tolist() if is_sbe else []

    feature_importances = pd.DataFrame(columns=X.columns.values, index=pd.Index([], name='ID'))
    model_scores: pd.DataFrame = None

    if is_sbe:
        test_scores = test_model_on_subset(predictor, X, y, best_model, subset=current_features)
        logger.info(f'Testing {len(current_features)=} -> {test_scores[metric]=}')
        model_scores = pd.DataFrame(test_scores, index=pd.Index([len(current_features)], name='ID'))

    for i in range(len(X.columns.values), 0, -1):
        logger.info(f'Ranking {i=} models when using {len(current_features)=} features')

        scores = algorithm_func(X, y, current_features,
                                predictor=predictor, model_name=best_model_name, model=best_model,
                                n_cpus=cpus, target_metric=metric, time_limit=time_limit)

        feature_importances.loc[len(scores)] = scores

        if not is_sbe:
            worst_or_best = draw_best(scores, top=True)
            current_features.append(worst_or_best)
            logger.info(f'Adding {worst_or_best=} to the current best features')

        if is_sbe:
            worst_or_best = draw_best(scores, top=False)
            current_features.remove(worst_or_best)
            logger.info(f'Removing {worst_or_best=} from current features')

        test_scores = test_model_on_subset(predictor, X, y, best_model, subset=current_features)
        logger.info(f'Testing {len(current_features)=} -> {test_scores[metric]=}')
        if model_scores is None:
            model_scores = pd.DataFrame(test_scores, index=pd.Index([len(current_features)], name='ID'))
        else:
            model_scores.loc[len(current_features)] = test_scores

        if not recursive:
            break

    feature_importances.to_csv(os.path.join(store_path, f'feature_importance.csv'))
    model_scores.to_csv(os.path.join(store_path, f'leaderboard.csv'))

@click.command(help='Rank the features using one of the greedy algorithms and strategies', context_settings={'show_default': True})
@click.option('--predictor-directory', type=str, required=True, help='working directory with the automl chosen model')
@click.option('--dataset-directory', type=str, required=True, help='directory where the finetune.csv file is located')
@click.option('--algorithm', type=click.Choice(get_all_ranking_algorithms(), case_sensitive=False), required=True, help='the ranking algorithm to be used')
@click.option('--metric', type=click.Choice(get_scorer_names(), case_sensitive=False), default='accuracy', help='evaluation metric')
@click.option('--cpus', type=int, default=get_default_cpu_number(), help='number of CPU cores to assign')
@click.option('--recursive', is_flag=True, help='whether to perform the algorithm recursively')
@click.option('--time-limit', type=int, default=60, help='time limit for the computation if using autogluon')
def main(predictor_directory, dataset_directory, algorithm, metric, cpus, recursive, time_limit):
    rank(predictor_directory, dataset_directory, algorithm, metric, cpus, recursive, time_limit)

if __name__ == '__main__':
    main()
