import numpy as np
from sklearn.datasets import load_svmlight_file
import os
import gc


class DatasetLoader:

    def __init__(self, datasets_folder='./datasets'):
        self.datasets_folder = datasets_folder
        self.all_files = sorted(os.listdir(datasets_folder))
        self.datasets = [f for f in self.all_files if not f.endswith('.t') and not f.endswith('.')]

    def get_dataset(self, dataset):
        if dataset not in self.datasets:
            raise ValueError(f"Dataset does not exist in {self.datasets_folder} folder!")
        file_path = os.path.join(self.datasets_folder, dataset)
        if f'{dataset}.t' in self.all_files:
            dataset_dict = process_two_files(
                file_path,
                os.path.join(self.datasets_folder, f'{dataset}.t')
            )
        else:
            dataset_dict = process_single_file(file_path)
        return dataset_dict

    def get_dataset_params(self, dataset):
        dataset_dict = self.get_dataset(dataset)
        info = {
            "observations": dataset_dict["observations"],
            "features": dataset_dict["features"],
        }
        del dataset_dict
        gc.collect()
        return info

    def list_datasets(self):
        for dataset in self.datasets:
            info = self.get_dataset_params(dataset)
            print(
                '#' * 40,
                f'Dataset: {dataset}',
                f'Observations: {info["observations"]}',
                f'Features: {info["features"]}',
                sep='\n'
            )


def load_datasets() -> dict:
    """
    Loads datasets ready for analysis.
    Each key in dict is a dict with dataset and has following properties
    - X_train, y_train
    - X_test, y_test
    - info
    - observations
    - features
    """
    datasets = {
        "a9a": process_two_files('datasets/a9a', 'datasets/a9a.t'),
        "a8a": process_two_files('datasets/a8a', 'datasets/a8a.t'),
        "a7a": process_two_files('datasets/a7a', 'datasets/a7a.t'),
        "a6a": process_two_files('datasets/a6a', 'datasets/a6a.t'),
        "a5a": process_two_files('datasets/a5a', 'datasets/a5a.t'),
        "a4a": process_two_files('datasets/a4a', 'datasets/a4a.t'),
        "real-sim": process_single_file('datasets/real-sim'),
        "skin_nonskin": process_single_file('datasets/skin_nonskin'),
        "mushrooms": process_single_file('datasets/mushrooms'),
        "australian": process_single_file('datasets/australian_scale'),
        "phishing": process_single_file('datasets/phishing')
    }
    return datasets


def process_two_files(path_train, path_test, info=""):
    X_train, y_train = load_svmlight_file(path_train)
    X_train = X_train.toarray()

    X_test, y_test = load_svmlight_file(path_test)
    X_test = X_test.toarray()

    dataset = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "info": info,
        "observations": X_train.shape[0],
        "features": X_train.shape[1],
    }

    return dataset


def process_single_file(path, info=""):
    X, y = load_svmlight_file(path)
    X = X.toarray()
    np.random.seed(42)
    np.random.shuffle(X)

    train_obs = int(X.shape[0] * 0.7)

    dataset = {
        "X_train": X[:train_obs],
        "y_train": y[:train_obs],
        "X_test": X[train_obs:],
        "y_test": y[train_obs:],
        "info": info,
        "observations": X[:train_obs].shape[0],
        "features": X[:train_obs].shape[1],
    }

    return dataset


if __name__ == "__main__":
    dl = DatasetLoader()
    dl.list_datasets()
