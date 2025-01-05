from DecTree import DecisionTreeClassifier, DecisionTreeRegressor
import os
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
class RandomForest:
    def __init__(self, max_depth=5, min_leaf_split=10, n_estimators=10, max_features="n", use_bootstrap=True, n_jobs=-1):
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.use_bootstrap = use_bootstrap
        if n_jobs == -1:
            self.n_jobs = os.cpu_count()
        elif n_jobs <= 0:
            raise ValueError("n_jobs должно быть больше 0 или равно -1 (используются все ядра)")
        else:
            self.n_jobs = n_jobs

        self.models = [self.class_simple_model(max_depth=max_depth, min_leaf_split=min_leaf_split)
                       for _ in range(self.n_estimators)]

    def fit(self, X, y):
        self.calc_max_features(X.shape[1])
        X, y = self._prepare_inputs(X, y)

        def train_single_model(model):
            X_bootstrap, y_bootstrap = self.bootstrap(X, y)
            selected_indices = self.subspace_indices(X.shape[1])
            trained_model = self._fit_model(model, X_bootstrap[:, selected_indices], y_bootstrap)
            return selected_indices, trained_model

        results = Parallel(n_jobs=self.n_jobs)(delayed(train_single_model)(m) for m in self.models)

        self.feature_indices, self.models = zip(*results)
        self.feature_indices = list(self.feature_indices)
        self.models = list(self.models)

    def _fit_model(self, model, X, y):
        model.fit(X, y)
        return model

    def subspace_indices(self, feature_count):
        return np.random.choice(feature_count, size=self.max_features, replace=False)

    def calc_max_features(self, feature_count):
        if self.max_features == "sqrt":
            self.max_features = int(np.sqrt(feature_count))
        elif self.max_features == "log2":
            self.max_features = int(np.log2(feature_count))
        else:
            self.max_features = feature_count

    def bootstrap(self, X, y):
        if self.use_bootstrap:
            n_samples = X.shape[0]
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            return X_bootstrap, y_bootstrap
        return X, y

    def subspace(self, X):
        selected_indices = np.random.choice(X.shape[1], size=self.max_features, replace=False)
        X_subspace = X[:, selected_indices]
        return X_subspace

    @staticmethod
    def _prepare_inputs(X, y=None):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(X, list):
            X = np.array(X)
        X = X.astype(float)

        if y is not None:
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y = y.values
            if isinstance(y, list):
                y = np.array(y)
            if pd.api.types.is_categorical_dtype(y):
                y = y.astype(int)
            y = y.astype(float)
            return X, y

        return X

class MyRandomForestClassifier(RandomForest):
    def __init__(self, max_depth=5, min_leaf_split=10, n_estimators=10, max_features="n", use_bootstrap=True, n_jobs=-1):
        self.class_simple_model = DecisionTreeClassifier
        super().__init__(max_depth, min_leaf_split, n_estimators, max_features, use_bootstrap, n_jobs)

    def predict(self, X):
        X = self._prepare_inputs(X)
        def predict_single_model(i):
            return self.models[i].predict(X[:, self.feature_indices[i]])

        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(predict_single_model)(i) for i in range(len(self.models))
        )
        predictions = np.array(predictions)

        final_predictions = []
        for i in range(predictions.shape[1]):
            uniq, counts = np.unique(predictions[:, i], return_counts=True)
            final_predictions.append(uniq[np.argmax(counts)])

        return np.array(final_predictions, dtype=int)


class MyRandomForestRegressor(RandomForest):
    def __init__(self, max_depth=5, min_leaf_split=10, n_estimators=10, max_features="n", use_bootstrap=True, n_jobs=-1):
        self.class_simple_model = DecisionTreeRegressor
        super().__init__(max_depth, min_leaf_split, n_estimators, max_features, use_bootstrap, n_jobs)

    def predict(self, X):
        X = self._prepare_inputs(X)
        def predict_single_model(i):
            return self.models[i].predict(X[:, self.feature_indices[i]])

        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(predict_single_model)(i) for i in range(len(self.models))
        )
        predictions = np.array(predictions)
        predictions = np.mean(predictions, axis=0)

        return predictions