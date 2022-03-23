from typing import Tuple

import numpy as np 

from one_hot_intuition import get_dataset
from one_hot_intuition import to_categorical

def softmax (y_pred: np.ndarray) -> np.ndarray:
    probabilities = np.zeros_like(y_pred)
    for i in range(len(y_pred)):
        exps= np.exp(y_pred[i])
        probabilities [i] = exps/np.sum(exps)
    return probabilities

if __name__ == "__main__":
    x,y = get_dataset()
    print(y.shape)
    print(y)

    y_categrical = to_categorical(y,num_classes=2)
    print(y_categrical.shape)
    print(y_categrical)

    y_prob = softmax(y_categrical)
    print(y_prob.shape)
    print(y_prob)