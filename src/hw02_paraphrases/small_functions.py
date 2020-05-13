import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def square_roots(start, end, length):
    """
    Returns a 1d numpy array of the specified length, containing the square roots of equi-distant input values
    between start and end (both included).

    >>> square_roots(4,9,3)
    array([2.        , 2.54950976, 3.        ])
    """
    # TODO: Exercise 2.1
    equidistant=(start+end)/2
    hilf_list=[start,equidistant,end]
    array=np.sqrt(hilf_list)
    return array[:length]
def odd_ones_squared(rows, cols):
    """
    Returns a 2d numpy array with shape (rows, cols). The matrix cells contain increasing numbers,
    where all odd numbers are squared.

    >>> odd_ones_squared(3,5)
    array([[  0,   1,   2,   9,   4],
           [ 25,   6,  49,   8,  81],
           [ 10, 121,  12, 169,  14]])
    """
    # TODO: Exercise 2.2
    array=np.arange(rows*cols).reshape(rows,cols)
    odd_list=[x for x in range(rows*cols) if x%2==1]
    for i in range(rows):
        for j in range(cols):
            if(array[i][j] in odd_list):
             array[i][j]*=array[i][j]
    return array