# Lucas Invernizzi Differentially Private Naive Bayes Classifier

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


def gaussian_probability(val, mean, std_dev):
    exponent = np.exp(-(np.power(val - mean, 2) / (2 * np.power(std_dev, 2))))
    return (1 / (np.sqrt(2 * np.pi) * std_dev)) * exponent


class NaiveBayes:
    def __init__(self, dataset=None, privacy=None, prior={}, data=None, labels=None, non_num_attr=None):
        self.dataset = dataset
        self.privacy = privacy
        self.prior = prior
        if data is None and labels is None and non_num_attr is None:
            self.data = pd.DataFrame()
            self.labels = pd.DataFrame()
            self.non_num_attr = {}
            self.load_data()
        else:
            self.data = data
            self.data.index = self.data.index + 1
            self.labels = labels
            self.non_num_attr = non_num_attr

    def load_data(self):
        x, y = fetch_openml(self.dataset, return_X_y=True)
        self.data = pd.DataFrame(x)
        self.labels = np.where(y == y[0], 1, 0)
        if self.dataset == 'lungcancer_GSE31210':
            self.non_num_attr[1] = 4  # 2nd attribute has 4 possible values
            self.non_num_attr[3] = 2  # 4th attribute has 2 possible values
        elif self.dataset == 'ilpd':
            self.non_num_attr[1] = 2  # 2nd attribute has 2 possible values

    def classify(self, row):
        prob = {}
        for label in range(2):  # for every possible label
            matching_class = self.labels == label  # truth vector of examples with this class label
            num_matching_class = np.sum(matching_class)  # number of examples with this class label
            prob[label] = self.prior[label]  # prior for this class label

            for attr, col in self.data.items():  # for every column (attribute) of training data
                if attr in self.non_num_attr.keys():  # if this attribute is categorical
                    m_val = self.non_num_attr[attr]  # set m=v

                    # truth vector: if this training data column has the test value for this attribute
                    matching_value = col == row[attr]

                    # truth vector: where the training data column equals the test value and this class label
                    matching_candv = np.logical_and(matching_class, matching_value)
                    num_matching_candv = np.sum(matching_candv)  # how many of above are true

                    if self.privacy is not None:  # privacy implementation
                        num_matching_candv += np.random.laplace(0, 1 / self.privacy)

                    # multiply the product with laplace smoothing:
                    prob[label] *= (num_matching_candv + 1) / (num_matching_class + m_val + 1)
                else:
                    # values of where the training data column has this class label
                    matching_class_examples = col[np.where(matching_class == True)[0]]
                    mean = np.mean(matching_class_examples)
                    std_dev = np.std(matching_class_examples)

                    if self.privacy is not None:  # privacy implementation
                        u = np.max(matching_class_examples)
                        l = np.min(matching_class_examples)
                        n = len(matching_class_examples)
                        s_mean = (u - l) / (n + 1)
                        s_std_dev = np.sqrt(n) * s_mean
                        sf_mean = s_mean / self.privacy
                        sf_std_dev = s_std_dev / self.privacy
                        mean += np.random.laplace(0, sf_mean)
                        std_dev += np.random.laplace(0, sf_std_dev)

                    # multiply the product with the gaussian probability
                    prob[label] *= gaussian_probability(row[attr], mean, std_dev)

        # chooses which label has the higher probability
        if prob[0] <= prob[1]:
            choice = 1
        else:
            choice = 0

        return choice

    def evaluate(self, test_data, test_labels):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for ex, row in test_data.iterrows():
            predicted_label = self.classify(row)
            actual_label = test_labels[ex]
            if predicted_label == 1:
                if actual_label == 1:
                    tp += 1
                elif actual_label == 0:
                    fp += 1
            elif predicted_label == 0:
                if actual_label == 0:
                    tn += 1
                elif actual_label == 1:
                    fn += 1
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
        return accuracy, precision, recall

    def test_model(self, folds=10):
        pos_indices = np.where(self.labels == 1)[0]  # indices of positive examples
        neg_indices = np.where(self.labels == 0)[0]  # indices of negative examples
        np.random.shuffle(pos_indices)  # randomize indices
        np.random.shuffle(neg_indices)
        pos_folds = np.array(np.array_split(pos_indices, folds))  # split into some number of folds
        neg_folds = np.array(np.array_split(neg_indices, folds))  # pos and neg separately for stratified CV
        fold_indices = np.arange(folds)
        num_pos_labels = np.sum(self.labels == 1)
        num_neg_labels = np.sum(self.labels == 0)
        if self.privacy is not None: # privacy addition for each class
            num_pos_labels += np.random.laplace(0, 1)
            num_neg_labels += np.random.laplace(0, 1)
        self.prior[1] = num_pos_labels / len(self.labels)
        self.prior[0] = num_neg_labels / len(self.labels)
        accuracies = []
        precisions = []
        recalls = []
        for fold in fold_indices:
            train_indices = np.where(fold_indices != fold)[0]  # all but one fold
            train = np.append(pos_folds[train_indices], neg_folds[train_indices]).flatten()  # combine pos and neg folds
            train = np.hstack(train).ravel()  # ravel into a single array of indices
            test = np.append(pos_folds[fold], neg_folds[fold]).flatten()
            nb = NaiveBayes(prior=self.prior,
                            privacy=self.privacy,
                            data=self.data.loc[train].reset_index(drop=True),
                            labels=self.labels[train],
                            non_num_attr=self.non_num_attr)
            acc, precision, recall = nb.evaluate(self.data.loc[test].reset_index(drop=True), self.labels[test])
            accuracies.append(acc)
            precisions.append(precision)
            recalls.append(recall)
        return accuracies


if __name__ == '__main__':
    datasets = ['diabetes', 'lungcancer_GSE31210', 'ilpd']
    privacies = [None, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    accuracies = {}
    avg_accuracy = {}
    std_dev = {}

    for dataset in datasets:
        print('New Dataset', dataset)

        if dataset not in accuracies.keys():
            accuracies[dataset] = {}
            avg_accuracy[dataset] = {}
            std_dev[dataset] = {}

            for privacy in privacies:
                if privacy not in accuracies[dataset].keys():
                    accuracies[dataset][privacy] = []

                nb = NaiveBayes(dataset=dataset, privacy=privacy)

                for i in range(5):
                    accuracies[dataset][privacy].append(nb.test_model())

                avg_accuracy[dataset][privacy] = np.mean(accuracies[dataset][privacy])
                std_dev[dataset][privacy] = np.std(accuracies[dataset][privacy])

        print('Accuracy for dataset ', dataset)
        print(avg_accuracy[dataset])
        print('Standard deviation for dataset', dataset)
        print(std_dev[dataset])
