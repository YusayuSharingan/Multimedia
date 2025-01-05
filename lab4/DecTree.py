import numpy as np
import pandas as pd

class Node:
    def __init__(self, threshold=None, idx_feature=None, criterion=None, parent=None, left_children=None, right_children=None, y_values=None):
        self.threshold = threshold
        self.idx_feature = idx_feature
        self.parent = parent
        self.left_children = left_children
        self.right_children = right_children
        self.y_values = y_values
        self.criterion = criterion

class DecisionTree:
    def __init__(self, max_depth=1000, min_leaf_split=2):
        self.root = None
        self.max_depth = max_depth
        self.min_leaf_split = min_leaf_split

    def fit(self, X, y):
        X, y = self._prepare_inputs(X, y)
        self.root = self.build_tree(X, y)

    def build_tree(self, X, y, parent=None, depth=0):
        if X.shape[0] <= self.min_leaf_split or depth >= self.max_depth:
            return Node(parent=parent, y_values=y)

        best_criterion, best_idx_feature, best_threshold = self.find_best_feature_split(X, y)

        if best_criterion is None or best_criterion <= 0:
            return Node(parent=parent, y_values=y)

        node = Node(threshold=best_threshold, idx_feature=best_idx_feature, criterion=best_criterion, parent=parent, y_values=y)

        left_mask = X[:, best_idx_feature] <= best_threshold
        right_mask = ~left_mask
        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]

        node.left_children = self.build_tree(left_X, left_y, parent=node, depth=depth + 1)
        node.right_children = self.build_tree(right_X, right_y, parent=node, depth=depth + 1)

        return node

    def predict(self, X) -> np.ndarray:
        """
        Прогнозирует метки классов для входных данных X.

        X: np.ndarray с признаками (n_samples, n_features).
        Возвращает: np.ndarray с предсказанными метками классов (n_samples,).
        """
        X = self._prepare_inputs(X)
        def traverse_batch(node, X_batch):
            if node.left_children is None and node.right_children is None:
                return np.full(X_batch.shape[0], self._predict_leaf(node), dtype=float)

            left_mask = X_batch[:, node.idx_feature] <= node.threshold
            right_mask = ~left_mask

            predictions = np.empty(X_batch.shape[0], dtype=float)
            if left_mask.any():
                predictions[left_mask] = traverse_batch(node.left_children, X_batch[left_mask])
            if right_mask.any():
                predictions[right_mask] = traverse_batch(node.right_children, X_batch[right_mask])
            return predictions

        return traverse_batch(self.root, X)

    def print_tree(self):
        """
        Отображает дерево решений в виде текста.
        """
        self._print_subtree(self.root)

    def _print_subtree(self, node, depth=0):
        if node is None:
            return
        indent = "  " * depth
        print(f"{indent}Node(threshold={node.threshold}, idx_feature={node.idx_feature}, criterion={node.criterion}, y_values={node.y_values})")
        self._print_subtree(node.left_children, depth + 1)
        self._print_subtree(node.right_children, depth + 1)


    def find_best_feature_split(self, X: np.ndarray, y: np.ndarray):
        best_idx_feature = None
        best_threshold = None
        best_criterion = 0

        for idx_feature in range(X.shape[1]):
            cur_criterion, cur_threshold = self.find_best_threshold_split(X[:, idx_feature], y)
            if cur_criterion > best_criterion:
                best_threshold = cur_threshold
                best_criterion = cur_criterion
                best_idx_feature = idx_feature

        return best_criterion, best_idx_feature, best_threshold

    def find_best_threshold_split(self, X_cols: np.ndarray, y: np.ndarray):
        best_criterion = 0
        best_threshold = None

        uniq_elem = np.unique(X_cols)
        for i in range(1, len(uniq_elem)):
            threshold = (uniq_elem[i-1] + uniq_elem[i]) / 2
            y_left, y_right = self.get_split(X_cols, y, threshold)
            cur_criterion = self.calculate_criterion(y, y_left, y_right)
            if cur_criterion > best_criterion:
                best_criterion = cur_criterion
                best_threshold = threshold

        return best_criterion, best_threshold

    def calculate_criterion(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray):
        total_len = len(y)
        len_left = len(y_left)
        len_right = len(y_right)

        if len_left == 0 or len_right == 0:
            return 0.0

        return (
            self.criterion(y)
            - (len_left / total_len * self.criterion(y_left)
            + len_right / total_len * self.criterion(y_right))
        )

    @staticmethod
    def get_split(X_cols, y, threshold):
        indices = X_cols <= threshold
        left = y[indices]
        right = y[~indices]
        return left, right

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



class DecisionTreeClassifier(DecisionTree):
    def __init__(self, criterion = "gini", max_depth=10, min_leaf_split=2):
        super().__init__(max_depth, min_leaf_split)
        self.criterion = self.calculate_gini
        if criterion == "entropy":
            self.criterion = self.calculate_entropy

    def _predict_leaf(self, node: Node):
        """
        Предсказание для листового узла.
        Возвращает класс, который чаще всего встречается в y_values узла.
        """
        unique, counts = np.unique(node.y_values, return_counts=True)
        return unique[np.argmax(counts)]

    @staticmethod
    def calculate_gini(y: np.ndarray):
        sum_counts = len(y)
        unique_elements, counts = np.unique(y, return_counts=True)
        p_k = counts / sum_counts
        return 1 - np.sum(p_k * p_k).item()

    @staticmethod
    def calculate_entropy(y: np.ndarray):
        sum_counts = len(y)
        unique_elements, counts = np.unique(y, return_counts=True)
        p_k = counts / sum_counts
        log_p_k = np.where(p_k > 0, np.log(p_k) / np.log(2), 0.0)
        return - np.sum(p_k * log_p_k).item()




class DecisionTreeRegressor(DecisionTree):
    def __init__(self, max_depth=10, min_leaf_split=2):
        super().__init__(max_depth, min_leaf_split)
        self.criterion = self.calculate_mse

    def _predict_leaf(self, node: Node):
        """
        Предсказание для листового узла.
        Возвращает класс, который чаще всего встречается в y_values узла.
        """
        if node.y_values.size == 0:
            return 0.0
        return node.y_values.mean()

    @staticmethod
    def calculate_mse(y: np.ndarray):
        if y.size == 0:
            return 0.0
        y_pred = np.mean(y)
        mse = np.mean((y - y_pred) ** 2)
        return mse