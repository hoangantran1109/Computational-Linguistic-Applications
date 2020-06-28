from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix


def read_word2vec_file(filename):
    """
    Reads a file in word2vec format. File Format:
    First line:
    <number_of_words> <number_of_dimensions>
    All other lines:
    <word> <values of vector components, white-space separated>

    Example:
    (2 words and 4 dimensions per word)

    2 4
    first 0 0.1 0.2 0.3
    second 1 2 3 4

    Returns:
        a tuple (m_word_vectors, word_to_id), where
        m_word_vectors -- Numpy array of size number_of_words x number_of_dimensions, containing the word vectors.
        word_to_id -- Dictionary, mapping each word to the corresponding row in the matrix.
    """
    with open(filename, "r") as modelfile:
        rows, cols = modelfile.readline().split(" ")
        vocab=[]
        features_list = []
        for line in modelfile.readlines():
            vocab.append((line.split(" ", 1))[0])
            _, features = line.split(" ", 1)
            features_list += [float(feature) for feature in features.split(" ")]
        final_vocab=list(dict.fromkeys(vocab))
        word_to_id = {w: i for i, w in enumerate(final_vocab)}
        #word_to_id = {w: i for i, w in enumerate(vocab)}
        m_word_vectors = np.array(features_list).reshape(int(rows), int(cols))
    # TODO Exercise 1 *Done

    return m_word_vectors, word_to_id


def read_entity_types_file(filename, m_word_vectors, word_to_id, type_to_id = None):
    """
    Reads a file containing entities with their types, and returns a feature matrix and a label matrix,
    and the mapping used to encode the types.
    The feature matrix is, for every instance, the average of the word vectors for the tokens in that instance.
    If there are no word vectors for the instance, a vector with all zeros is used.
    The label matrix is, for every instance, the 0-1 encoding of the types for this instance.

    If a mapping from types to ids (columns in the label matrix) is provided, this is used; otherwise it is created.

    Every input line has the following form:
    <Entity>\t<types, white-space separated>
    Keyword arguments:
        filename -- The name of the (train, dev, or test) file.
        m_word_vectors -- Matrix with word vectors, which is used to create the feature representations.
        word_to_id -- dictionary, mapping word strings to their row in m_word_vectors
        type_to_id -- dictionary, mapping types to their column in the label matrix (which is created)
    Returns:
        Tuple of (x_features, y_labels, type_to_id)
        x_features -- Feature matrix. Numpy array of size number_of_instances x number_of_dimensions.
        y_labels -- Label matrix with values 0 or 1. Scipy sparse matrix of size number_of_instances x number_of_types.
        type_to_id -- dictionary, mapping types to their column in y_labels.
    """
    if type_to_id is None:
        grow_type_dict = True
        type_to_id = dict()
    else:
        grow_type_dict = False
    num_dims = m_word_vectors.shape[1]

    feature_rows = []
    # Rows, columns and 1-0 encoding for label matrix in coordinate format.
    label_rows = []
    label_cols = []
    label_values = []

    current_row = 0
    with open(filename, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            token_ids = [word_to_id[t] for t in parts[0].split(" ") if t in word_to_id]
            num_tokens = len(token_ids)
            feature_vec_sum = 0
            if num_tokens > 0:
                for token_id in token_ids:
                    feature_vec_sum += m_word_vectors[token_id]
                avg_feature_vec = (1 / num_tokens) * feature_vec_sum
            else:
                avg_feature_vec = np.zeros(num_dims)
            # TODO: Exercise 2 *Done
            feature_rows.append(avg_feature_vec)
            for type in parts[1].split(" "):
                if grow_type_dict and type not in type_to_id:
                    type_to_id[type] = len(type_to_id)
                if type in type_to_id:
                    label_rows.append(current_row)
                    label_cols.append(type_to_id[type])
                    label_values.append(1)
            current_row += 1
    x_features = np.vstack(feature_rows)
    y_labels = coo_matrix((label_values, (label_rows, label_cols)), shape=(current_row, len(type_to_id)))
    return x_features, y_labels.tocsr(), type_to_id
