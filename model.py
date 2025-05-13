from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import joblib
import pandas as pd
import numpy as np
from typing import List
import json


class LogisticRegressionScratch:
    def __init__(self, C: float = 1.0, lr: float = 0.01, n_iter: int = 1000):
        """
        C     : inverse regularization strength (bigger C => less reg)
        lr    : learning rate
        n_iter: number of gradient steps
        """
        self.C = C
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        # lambda = 1/C
        lambda_param = 1.0 / self.C

        for _ in range(self.n_iter):
            # linear model
            linear = X.dot(self.weights) + self.bias
            # sigmoid
            y_pred = 1.0 / (1.0 + np.exp(-linear))

            # gradients w/ L2 regularization
            dw = (1.0 / n_samples) * X.T.dot(y_pred - y) \
                 + lambda_param * self.weights
            db = (1.0 / n_samples) * np.sum(y_pred - y)

            # update
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        linear = X.dot(self.weights) + self.bias
        probs  = 1.0 / (1.0 + np.exp(-linear))
        return np.vstack([1-probs, probs]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:,1] >= 0.5).astype(int)


class SVMScratch:
    def __init__(self,
                 C: float = 1.0,
                 lr: float = 0.001,
                 n_iter: int = 1000,
                 kernel: str = 'linear',
                 gamma: str = 'scale'):
        """
        C      : weight on hinge-loss term (larger C => less slack)
        lr     : learning rate
        n_iter : passes over data
        kernel : only 'linear' implemented
        gamma  : placeholder (not used for linear)
        """
        self.C = C
        self.lr = lr
        self.n_iter = n_iter
        self.kernel = kernel
        self.gamma = gamma
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # only linear supported
        if self.kernel != 'linear':
            raise NotImplementedError("SVMScratch only supports kernel='linear'")

        # map {0,1}→{-1,1}
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        # init
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                margin = y_[idx] * (x_i.dot(self.w) - self.b)
                if margin >= 1:
                    # only regularization gradient
                    dw = self.w
                    db = 0.0
                else:
                    # reg + hinge
                    dw = self.w - self.C * y_[idx] * x_i
                    db = -self.C * y_[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.w) - self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # sigmoid of the decision function
        df = self.decision_function(X)
        probs = 1.0 / (1.0 + np.exp(-df))
        return np.vstack([1-probs, probs]).T    

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # for leaf nodes


class DecisionTreeScratch:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self.max_depth
            or n_samples < self.min_samples_split
            or num_labels == 1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        best_feat, best_thresh = self._best_criteria(X, y, n_features)
        if best_feat is None:
            return Node(value=self._most_common_label(y))

        left_idx, right_idx = self._split(X[:, best_feat], best_thresh)
        left  = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, n_features):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left_idx, right_idx = self._split(X[:, feat], t)
                # **skip** thresholds that put everything on one side
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                gain = self._information_gain(y, X[:, feat], t)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = t
        return split_idx, split_thresh

    def _information_gain(self, y, X_col, split_thresh):
        # parent Gini
        parent_gini = self._gini(y)
        left_idx, right_idx = self._split(X_col, split_thresh)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        gini_l = self._gini(y[left_idx])
        gini_r = self._gini(y[right_idx])
        # weighted
        child_gini = (n_l/n) * gini_l + (n_r/n) * gini_r
        return parent_gini - child_gini

    def _gini(self, y):
        counts = np.bincount(y)
        ps = counts / counts.sum()
        return 1 - np.sum(ps**2)

    def _split(self, X_col, split_thresh):
        left_idx  = np.where(X_col <= split_thresh)[0]
        right_idx = np.where(X_col >  split_thresh)[0]
        return left_idx, right_idx

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse(x, self.root) for x in X])
    
class RandomForestScratch:
    def __init__(self,
                 n_estimators=100,
                 max_depth=10,
                 min_samples_split=2,
                 max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees: List[DecisionTreeScratch] = []
        self.features_idxs: List[np.ndarray] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        y = np.asarray(y)
        n_samples, n_features = X.shape
        if self.max_features == 'sqrt':
            max_feats = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_feats = int(np.log2(n_features))
        else:
            max_feats = n_features

        for _ in range(self.n_estimators):
            # bootstrap sample
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_samp, y_samp = X[idxs], y[idxs]
            feat_idxs = np.random.choice(n_features, max_feats, replace=False)

            tree = DecisionTreeScratch(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth
            )
            # train on subset of features
            tree.fit(X_samp[:, feat_idxs], y_samp)
            self.trees.append(tree)
            self.features_idxs.append(feat_idxs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # collect each tree's predictions
        all_preds = np.array([
            tree.predict(X[:, fi]) for tree, fi in zip(self.trees, self.features_idxs)
        ])
        # majority vote
        return np.apply_along_axis(lambda row: np.bincount(row).argmax(),
                                   axis=0,
                                   arr=all_preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # fraction of trees voting class 1
        all_votes = np.array([
            tree.predict(X[:, fi]) for tree, fi in zip(self.trees, self.features_idxs)
        ])
        proba1 = all_votes.mean(axis=0)
        return np.vstack([1 - proba1, proba1]).T
    
class DecisionTreeRegressorScratch:
    def __init__(self, min_samples_split=2, max_depth=3):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if (depth >= self.max_depth) or (n_samples < self.min_samples_split):
            leaf_val = y.mean()
            return Node(value=leaf_val)

        best_feat, best_thresh = None, None
        best_loss = float('inf')

        for feat in range(n_features):
            for t in np.unique(X[:, feat]):
                left_idx, right_idx = np.where(X[:, feat] <= t)[0], np.where(X[:, feat] > t)[0]
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                loss = (
                    len(left_idx) * y[left_idx].var() +
                    len(right_idx)* y[right_idx].var()
                ) / n_samples
                if loss < best_loss:
                    best_loss, best_feat, best_thresh = loss, feat, t

        if best_feat is None:
            return Node(value=y.mean())

        left_idx, right_idx = np.where(X[:, best_feat] <= best_thresh)[0], np.where(X[:, best_feat] > best_thresh)[0]
        left = self._grow_tree(X[left_idx], y[left_idx], depth+1)
        right= self._grow_tree(X[right_idx], y[right_idx], depth+1)
        return Node(best_feat, best_thresh, left, right)

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse(x, self.root) for x in X])


class XGBoostScratch:
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 max_depth=3,
                 min_samples_split=2):
        self.n_estimators      = n_estimators
        self.lr                = learning_rate
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.trees: List[DecisionTreeRegressorScratch] = []
        self.init_score = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # initialize log-odds
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        p = np.clip(y.mean(), 1e-5, 1 - 1e-5)
        self.init_score = np.log(p / (1 - p))
        y_pred = np.full_like(y, self.init_score, dtype=float)

        for _ in range(self.n_estimators):
            # pseudo-residuals
            pred_proba = 1 / (1 + np.exp(-y_pred))
            grad = y - pred_proba

            tree = DecisionTreeRegressorScratch(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth
            )
            tree.fit(X, grad)
            self.trees.append(tree)

            update = tree.predict(X)
            y_pred += self.lr * update

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.full(X.shape[0], self.init_score, dtype=float)
        for tree in self.trees:
            y_pred += self.lr * tree.predict(X)
        probs = 1 / (1 + np.exp(-y_pred))
        return np.vstack([1 - probs, probs]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
    



class DraftPredictor:
    def __init__(
        self,
        champion_info_path: str,
        model_type: str = 'xgboost',
        # shared hyper-params
        lr: float = 0.01,
        n_iter: int = 1000,
        C: float = 1.0,
        kernel: str = 'linear',
        gamma: str = 'scale',
        # RF / XGBoost
        n_estimators: int = 100,
        max_depth: int = None,
        learning_rate: float = 0.1,
    ):
        mt = model_type.lower()

        if mt == 'logistic_regression':
            # our scratch logistic takes C, lr, n_iter
            self.model = LogisticRegressionScratch(
                C=C,
                lr=lr,
                n_iter=n_iter
            )

        elif mt == 'svm':
            # scratch SVM takes C, lr, n_iter, kernel, gamma
            self.model = SVMScratch(
                C=C,
                lr=lr,
                n_iter=n_iter,
                kernel=kernel,
                gamma=gamma
            )

        elif mt == 'random_forest':
            self.model = RandomForestScratch(
                n_estimators=n_estimators,
                max_depth=max_depth
            )

        elif mt == 'xgboost':
            self.model = XGBoostScratch(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate
            )

        else:
            raise ValueError(f"Unknown model_type '{model_type}'")

        # load champion info
        with open(champion_info_path, 'r') as f:
            data = json.load(f)["data"]

        self.champion_info = data
        self.id_to_name = {
            int(champ["id"]): champ["key"]
            for champ in data.values()
            if champ["key"] != "None"
        }

    def train(self, X: np.ndarray, y: np.ndarray):
        print(f"Training {self.model.__class__.__name__} on {X.shape[0]} samples, {X.shape[1]} features")
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        joblib.dump({'model': self.model}, path)

    def load(self, path: str):
        state = joblib.load(path)
        self.model = state['model']