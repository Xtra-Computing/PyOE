# refer from https://github.com/messaoudia/AdaptiveRandomForest

from collections import defaultdict, Counter
import numpy as np

from skmultiflow.rules.numeric_attribute_class_observer import GaussianNumericAttributeClassObserver
from skmultiflow.rules.nominal_attribute_class_observer import NominalAttributeClassObserver
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.bayes import do_naive_bayes_prediction
from skmultiflow.trees.hoeffding_tree import HoeffdingTreeClassifier
import random

class ARFHoeffdingTree(HoeffdingTreeClassifier):
    """
    ARFHoeffding Tree
    A Hoeffding tree is an incremental, anytime decision tree induction algorithm that is capable of learning from
    massive data streams, assuming that the distribution generating examples does not change over time. Hoeffding trees
    exploit the fact that a small sample can often be enough to choose an optimal splitting attribute. This idea is
    supported mathematically by the Hoeffding bound, which quantifies the number of observations (in our case, examples)
    needed to estimate some statistics within a prescribed precision (in our case, the goodness of an attribute).
    A theoretically appealing feature of Hoeffding Trees not shared by other incremental decision tree learners is that
    it has sound guarantees of performance. Using the Hoeffding bound one can show that its output is asymptotically
    nearly identical to that of a non-incremental learner using infinitely many examples.
    ARFHoeffding tree is based on Hoeffding tree and it has two main differences. Whenever a new node is created, a
    subset of m random attributes is chosen and split attempts are limited to that subset.
    Second difference is that there is no early tree prunning.
        See for details:
        G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams.
        In KDD’01, pages 97–106, San Francisco, CA, 2001. ACM Press.
        Implementation based on:
        Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer (2010);
        MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604
    Parameters
    ----------
    m: Int
        Number of random attributes for split on each node
    grace_period: Int
        The number of instances a leaf should observe between split attempts.
    delta_w: float
        Warning threshold of change detection for ADWIN change detector
    delta_d: float
        Change threshold of change detection for ADWIN change detector
    no_pre_prune: Boolean
        If True, disable pre-pruning. Default: True
    leaf_prediction: String
        Prediction mechanism used at leafs.
        'mc' - Majority Class
        'nb' - Naive Bayes
        'nba' - Naive BAyes Adaptive
    Other attributes for Hoeffding Tree:
    HoeffdingTree.max_byte_size: Int
        Maximum memory consumed by the tree.
    HoeffdingTree.memory_estimate_period: Int
        How many instances between memory consumption checks.
    HoeffdingTree.split_criterion: String
        Split criterion to use.
        'gini' - Gini
        'info_gain' - Information Gain
    HoeffdingTree.split_confidence: Float
        Allowed error in split decision, a value closer to 0 takes longer to decide.
    HoeffdingTree.tie_threshold: Float
        Threshold below which a split will be forced to break ties.
    HoeffdingTree.binary_split: Boolean
        If True only allow binary splits.
    HoeffdingTree.stop_mem_management: Boolean
        If True, stop growing as soon as memory limit is hit.
    HoeffdingTree.remove_poor_atts: Boolean
        If True, disable poor attributes.
    HoeffdingTree.nb_threshold: Int
        The number of instances a leaf should observe before permitting Naive Bayes.
    HoeffdingTree.nominal_attributes: List
        List of Nominal attributes
    """
    def __init__(self, m, delta_w, delta_d, grace_period=50, leaf_prediction='nb', no_pre_prune=True):
        super().__init__()
        self.m = m
        self.remove_poor_atts = None
        self.no_preprune = no_pre_prune
        self.delta_warning = delta_w
        self.delta_drift = delta_d
        self.adwin_warning = ADWIN(delta=self.delta_warning)
        self.adwin_drift = ADWIN(delta=self.delta_drift)
        self.leaf_prediction = leaf_prediction
        self.grace_period = grace_period

    @staticmethod
    def is_randomizable():
        return True

    def rf_tree_train(self, X, y):
        """
        This function calculates Poisson(6) and assigns this as a weight of instance.
        If Poisson(6) returns zero, it doesn't use this instance for training.
        :param X: Array
            Input vector
        :param y: Array
            True value of class for X
        """
        w = np.random.poisson(6)
        if w > 0:
            self.partial_fit([X], [y.item()], sample_weight=[w])


class AdaptiveRandomForest:
    """ AdaptiveRandomForest or ARF
        An Adaptive Random Forest is a classification algorithm that want to make
        Random Forest, which is not a stream algorithm, be again among the best classifier in streaming
        In this code you will find the implementation of the ARF described on :
            Adaptive random forests for evolving data stream classification
            Heitor M. Gomes, Albert Bifet, Jesse Read, Jean Paul Barddal,
            Fabricio Enembreck, Bernhard Pfharinger, Geoff Holmes, Talel Abdessalem
        Parameters
        ----------
        nb_features: Int
            The number of features a leaf should observe.
        nb_trees: Int
            The number of trees that the forest should contain
        predict_method: String
            Prediction method: either Majority Classifier "mc", Average "avg"
        """

    def __init__(self, nb_features=5, nb_trees=100, predict_method="mc", pretrain_size=1000, delta_w=0.01, delta_d=0.001):
        """
        Constructor
        :param predict_method:
        :type predict_method:
        :param nb_features: maximum feature evaluated per split
        :param nb_trees: total number of trees
        """
        self.m = nb_features
        self.n = nb_trees
        self.predict_method = predict_method
        self.pretrain_size = pretrain_size
        self.delta_d = delta_d
        self.delta_w = delta_w

        self.Trees = self.create_trees()
        self.Weights = self.init_weights()
        self.B = defaultdict()
        self.number_of_instances_seen = 0

    def create_trees(self):
        """
        Create nb_trees, trees
        :return: a dictionnary of trees
        :rtype: Dictionnary
        """
        trees = defaultdict(lambda: ARFHoeffdingTree(self.m, self.delta_w, self.delta_d))
        for i in range(self.n):
            trees[i] = self.create_tree()
        return trees

    def create_tree(self):
        """
        Create a ARF Hoeffding tree
        :return: a tree
        :rtype: ARFHoeffdingTree
        """
        return ARFHoeffdingTree(self.m,self.delta_w, self.delta_d)

    def init_weights(self):
        """
        Init weight of the trees. Weight is 1 per default
        :return: a dictionnary of weight, where each weight is associated to 1 ARF Hoeffding Tree
        :rtype: Dictionnary
        """
        l = list()
        l.append(1)
        l.append(1)
        return defaultdict(lambda: l)

    def learning_performance(self, idx, y_predicted, y):
        """
        Compute the learning performance of one tree at the index "idx"
        :param idx: index of the tree in the dictionnary
        :type idx: Int
        :param y_predicted: Prediction result
        :type y_predicted: Int
        :param y: The real y, from the training
        :type y: Int
        :return: /
        :rtype: /
        """
        # if well predicted, count th
        if y == y_predicted[0]:
            self.Weights[idx][0] += 1

        self.Weights[idx][1] += 1

    def partial_fit(self, X, y, classes=None):
        """
        Partial fit over X and y arrays
        :param X: Features
        :type X: Numpy.ndarray of shape (n_samples, n_features)
        :param y: Classes
        :type y: Vector
        :return:
        :rtype:
        """
        new_tree = list()
        index_to_replace = list()
        rows, cols = X.shape

        for stream in range(rows):
            X_ = X[stream, :]
            y_ = y[stream]
            self.number_of_instances_seen += 1

            # first tree => idx = 0, second tree => idx = 1 ...

            for key, tree in self.Trees.items():
                if self.number_of_instances_seen > self.pretrain_size:
                    y_predicted = tree.predict(np.asarray(X_.unsqueeze(0)))
                    self.learning_performance(idx=key, y_predicted=y_predicted, y=y_)
                    if y_ == y_predicted[0]:
                        correct_prediction = 1
                    else:
                        correct_prediction = 0
                    tree.adwin_warning.add_element(correct_prediction)
                    tree.adwin_drift.add_element(correct_prediction)
                    if tree.adwin_warning.detected_change():
                        if self.B.get(key, None) is None:
                            b = self.create_tree()
                            self.B[key] = b
                    else:
                        if self.B.get(key, None) is not None:
                            self.B.pop(key)

                    if tree.adwin_drift.detected_change():
                        if self.B.get(key, None) is None:
                            # Added condition, there is some problem here, we detected a drift before warning
                            b = self.create_tree()  # Also too many trees is being created
                            self.B[key] = b
                        new_tree.append(self.B[key])
                        index_to_replace.append(key)
                tree.rf_tree_train(np.asarray(X_), np.asarray(y_))

            for key, value in self.B.items():
                value.rf_tree_train(np.asarray(X_), np.asarray(y_))  # Changed

            # tree ← B(tree)
            for key, index in enumerate(index_to_replace):
                self.Trees[index] = new_tree[key]
                self.B.pop(index)
                self.Weights[index][0] = 1
                self.Weights[index][1] = 1

            new_tree.clear()
            index_to_replace.clear()

    def predict(self, X):
        """
        Predicts the label of the X instance(s)
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
        All the samples we want to predict the label for.
        Returns
        -------
        list
        A list containing the predicted labels for all instances in X.
        """
        r, _ = X.shape
        predictions_result = list()

        for row in range(r):
            X_ = X[row]

            best_class = -1
            # average weight
            predictions = defaultdict(float)
            predictions_count = defaultdict(int)

            if self.predict_method == "avg":

                global_weight = 0.0

                for key, tree in self.Trees.items():
                    y_predicted = tree.predict([np.asarray(X_)])
                    learning_perf = self.Weights[key][0] / self.Weights[key][1]
                    predictions[y_predicted[0]] += learning_perf
                    global_weight += learning_perf
                    # predictions_count[y_predicted[0]] += 1

                max_weight = -1.0
                for key, weight in predictions.items():
                    w = predictions[key] / global_weight
                    if best_class != key and w > max_weight:
                        max_weight = w
                        best_class = key

            elif self.predict_method == "mc":
                for key, tree in self.Trees.items():
                    y_predicted = tree.predict([np.asarray(X_)])
                    predictions_count[y_predicted[0]] += 1
                max_value = -1.0

                for key, value in predictions_count.items():
                    if value > max_value:
                        best_class = key
                        max_value = value

            predictions_result.append(best_class)

        return predictions_result
