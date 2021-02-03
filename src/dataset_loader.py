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
            if dataset.startswith('a') and dataset.endswith('a'):
                n_features = 123
            dataset_dict = process_two_files(
                file_path,
                os.path.join(self.datasets_folder, f'{dataset}.t'),
                n_features=n_features
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


def process_two_files(path_train, path_test, n_features=-1, info=""):
    X_train, y_train = load_svmlight_file(path_train)
    X_train = X_train.toarray()

    if X_train.shape[1] < n_features:
        X_tmp_copy = X_train
        X_train = np.zeros((X_tmp_copy.shape[0], n_features))
        X_train[:, :X_tmp_copy.shape[1]] = X_tmp_copy
        del X_tmp_copy

    X_test, y_test = load_svmlight_file(path_test)
    X_test = X_test.toarray()

    if X_test.shape[1] < n_features:
        X_tmp_copy = X_test
        X_test = np.zeros((X_tmp_copy.shape[0], n_features))
        X_test[:, :X_tmp_copy.shape[1]] = X_tmp_copy
        del X_tmp_copy

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
