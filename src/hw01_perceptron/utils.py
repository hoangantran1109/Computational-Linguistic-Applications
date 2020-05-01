import random

from collections import defaultdict


from nltk import word_tokenize


def dot(dictA, dictB):
    return sum([dictA.get(tok) * dictB.get(tok, 0) for tok in dictA]) # TODO: Ex. 2: return vector product between features vectors represented by dictA and dictB.

def normalized_tokens(text):
    return [token.lower() for token in word_tokenize(text)]

class DataInstance:
    def __init__(self, feature_counts, label):
        """ A data instance consists of a dictionary with feature counts (string -> int) and a label (True or False)."""
        self.feature_counts = feature_counts
        self.label = label

    @classmethod
    def from_list_of_feature_occurrences(cls, feature_list, label):
        """ Creates feature counts for all features in the list."""
        feature_counts = dict()
        # TODO: Ex. 3: create a dictionary that contains for each feature in the list the count how often it occurs.
        for feature in feature_list:
            count = feature_counts.get(feature, 0)
            feature_counts[feature] = count + 1
        return cls(feature_counts, label)


    @classmethod
    def from_text_file(cls, filename, label):
        with open(filename, 'r') as myfile:
            token_list = normalized_tokens(myfile.read().strip())
        return cls.from_list_of_feature_occurrences(token_list, label)


class Dataset:
    def __init__(self, instance_list):
        """ A data set is defined by a list of instances """
        self.instance_list = instance_list
        self.feature_set = set.union(*[set(inst.feature_counts.keys()) for inst in instance_list])


    def get_topn_features(self, n):
        """ This returns a set with the n most frequently occurring features (i.e. the features that are contained in most instances)."""

        word_to_count = defaultdict(int)

        for inst in self.instance_list:
            for feature, count in inst.feature_counts.items():
                word_to_count[(feature)] += count
        list=sorted(word_to_count,key=word_to_count.get,reverse=True)
        return set(list[0:n]) # TODO: Ex. 4: Return set of features that occur in most instances.

    def set_feature_set(self, feature_set):
        """
        This restrics the feature set. Only features in the specified set all retained. All other feature are removed
        from all instances in the dataset AND from the feature set."""
        # TODO: Ex. 5: Filter features according to feature set.

        for instance in self.instance_list:
            rausliste = []
            for feature in instance.feature_counts:
                if feature not in feature_set:
                    rausliste.append(feature)

            for raus_element in rausliste:
                instance.feature_counts.pop(raus_element)



    def most_frequent_sense_accuracy(self):
        """ Computes the accuracy of always predicting the overall most frequent sense for all instances in the dataset. """
        # x=0
        # total=0
        # top=self.get_topn_features(1)
        # set_f=self.set_feature_set(top)
        # for instance in self.instance_list:
        #     if(instance.label==True):
        #         for value in instance.feature_counts.values():
        #             x=x+value
        #             total=total+value
        #     if (instance.label == False):
        #         for value in instance.feature_counts.values():
        #             total=total+value

        total1=0
        total_richtig=0
        for instance in self.instance_list:
            total1+=1
            if(instance.label):
                total_richtig+=1
        return 1-(total_richtig/total1)
        #return x/total # TODO: Ex. 6: Return accuracy of always predicting most frequent label in data set.

    def shuffle(self):
        """ Shuffles the dataset. Beneficial for some learning algorithms."""
        random.shuffle(self.instance_list)
