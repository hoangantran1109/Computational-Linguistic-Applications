import copy
import json

from hw01_perceptron.utils import dot

class PerceptronClassifier:
    def __init__(self, weights):
        # string to int
        self.weights = weights

    @classmethod
    def from_file(cls, filename):
        """
        Load model file and construct PerceptronClassifier.
        """
        with open(filename, 'r') as modelfile:
            weights = json.load(modelfile)
        return cls(weights)

    @classmethod
    def for_dataset(cls, dataset):
        """
        Initialize PerceptronClassifier for dataset. A classifier that
        is constructed with this method still needs to be trained..
        """
        weights = {w:0 for w in dataset.feature_set}
        return cls(weights)

    def prediction(self, counts):
        """
        Return True if prediction for counts is ham, False if prediction is spam
        counts: Bag of words representation of email
        """
        return dot(counts, self.weights) > 0

    def update(self, instance):
        """
        Perform perceptron update, if the wrong label is predicted.
        Return a boolean value indicating whether an update was performed.
        """
        predicted_output = self.prediction(instance.feature_counts)
        error = 0 # TODO: Ex. 7: Replace 0 with the correct calculation of the error
        if not instance.label and predicted_output :
            error = 1
        elif  instance.label and not predicted_output :
            error = -1
        do_update = error !=0
        if do_update:
            # TODO: Ex. 7: Replace pass with update of feature weights
            for feature, count in instance.feature_counts.items():
              self.weights[feature]-=error*count
        return do_update

    def training_iteration(self, dataset):
        """
        Iterate over each instance of dataset and perform perceptron update.
        Return number of updates that were performed (number of train errors).
        """
        dataset.shuffle()
        for instance in dataset.instance_list:
            self.update(instance)

    def train(self, training_set, development_set, iterations):
        """
        Train classifier and return best development accuracy.
        """
        best_dev_accuracy = 0.0
        best_weights = self.weights
        for i in range(iterations):
            self.training_iteration(training_set)
            train_accuracy = self.prediction_accuracy(training_set)
            development_accuracy = self.prediction_accuracy(development_set)
            if development_accuracy > best_dev_accuracy:
                best_dev_accuracy = development_accuracy
                best_weights = self.weights.copy()
            print("Iteration: %d \t Train Accuracy: %.4f \t Dev Accuracy: %.4f \t Best Dev Accuracy: %.4f" % (i, train_accuracy, development_accuracy, best_dev_accuracy))
        self.weights = best_weights
        return best_dev_accuracy

    def prediction_accuracy(self, dataset):
        """
        Caclculate accuracy of classifier on labelled dataset.
        """
        num_errors = 0
        for instance in dataset.instance_list:
            if self.prediction(instance.feature_counts) != instance.label:
                num_errors += 1
        return 1 - num_errors / len(dataset.instance_list)


    def prediction_f_measure(self, dataset, for_label):
        """
        Caclculate f_measure of classifier for a labelled dataset and a specified label.
        """
        tp=0
        tn=0
        fp=0
        fn=0

        for instance in dataset.instance_list:
            vorher = (self.prediction(instance.feature_counts) == instance.label)!=for_label
            gold = instance.label==for_label
            if vorher and gold:
                tp+=1
            elif not vorher and  gold:
                fn+=1
            elif not vorher and gold:
                tn+=1
            elif not vorher and not gold:
                fp+=1

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        fmeasure = (2 * prec * rec) / (prec + rec)

        return fmeasure # TODO: Do the prediction for a given data set, and return the f-measure for a label of interest.

    def copy(self):
        """
        Return a copy of weights.
        """
        return PerceptronClassifier(copy.copy(self.weights))

    def features_for_class(self, is_positive_class, topn=10):
        """
        Determine the topn best features for a label (True or False).
        is_positive_class: can be True or False
        """
        high_to_low = True if is_positive_class else False
        return sorted(self.weights.items(), key=lambda x: x[1], reverse=high_to_low)[:topn]

    def save(self, filename):
        """
        Save model weights as JSON file.
        """
        with open(filename, 'w') as modelfile:
            json.dump(self.weights, modelfile)