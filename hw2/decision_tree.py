import numpy as np
from collections import Counter

def find_best_split(feature_vector, target_vector):
    feature_vector = np.asarray(feature_vector)
    target_vector = np.asarray(target_vector)

    sorted_indices = np.argsort(feature_vector)
    X_sorted = feature_vector[sorted_indices]
    y_sorted = target_vector[sorted_indices]

    thresholds_mask = X_sorted[1:] != X_sorted[:-1]
    thresholds = (X_sorted[1:] + X_sorted[:-1]) / 2
    thresholds = thresholds[thresholds_mask]

    if len(thresholds) == 0:
        return np.array([]), np.array([]), None, None

    total = len(y_sorted)
    ones_total = np.sum(y_sorted)

    y_cumsum = np.cumsum(y_sorted)[:-1]
    left_counts = np.arange(1, total)
    right_counts = total - left_counts

    y_cumsum = y_cumsum[thresholds_mask]
    left_counts = left_counts[thresholds_mask]
    right_counts = right_counts[thresholds_mask]

    p1_left = y_cumsum / left_counts
    p0_left = 1 - p1_left
    H_left = 1 - p1_left**2 - p0_left**2

    p1_right = (ones_total - y_cumsum) / right_counts
    p0_right = 1 - p1_right
    H_right = 1 - p1_right**2 - p0_right**2

    ginis = - (left_counts / total) * H_left - (right_counts / total) * H_right

    best_idx = np.argmin(ginis)
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, float("inf"), None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature].astype(float)
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {key: clicks.get(key, 0) / counts[key] if counts[key] > 0 else 0 for key in counts}
                sorted_categories = [cat for cat, _ in sorted(ratio.items(), key=lambda x: x[1])]
                categories_map = {cat: i for i, cat in enumerate(sorted_categories)}
                feature_vector = np.array([categories_map.get(x, 0) for x in sub_X[:, feature]])
            else:
                raise ValueError("Unknown feature type")

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini is None:
                continue

            temp_split = feature_vector < threshold
            left_size = np.sum(temp_split)
            right_size = len(temp_split) - left_size

            if self._min_samples_leaf is not None and (
                left_size < self._min_samples_leaf or right_size < self._min_samples_leaf
            ):
                continue

            if gini < gini_best:
                feature_best = feature
                gini_best = gini
                split = temp_split
                if feature_type == "real":
                    threshold_best = threshold
                else:
                    threshold_best = [cat for cat, idx in categories_map.items() if idx < threshold]

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        else:
            node["categories_split"] = threshold_best

        node["left_child"], node["right_child"] = {}, {}

        feature_type = self._feature_types[feature_best]
        feature_vector = sub_X[:, feature_best]
        
        if feature_type == "real":
            split = feature_vector.astype(float) < threshold_best
        else:
            split = np.isin(feature_vector, threshold_best)

        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        feature_type = self._feature_types[feature]

        if feature_type == "real":
            if float(x[feature]) < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError("Unknown feature type")
    
    def fit(self, X, y):
        self._tree = {}
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])
