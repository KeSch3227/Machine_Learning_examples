from typing import Tuple

import numpy as np 

from softmax_intuition import get_dataset
from softmax_intuition import softmax
from softmax_intuition import to_categorical

def cross_entropy(y_true:np.ndarray,y_pred: np.ndarray) ->float:
    num_samples = y_true.shape[0]
    loss = -np.sum(y_true*np.log(y_pred))/num_samples
    return loss 

if __name__ =="__main__":
    x,y = get_dataset()
    print(y)

    y_true = to_categorical(y,num_classes=2)
    print(y_true)

    y_logits = np.array([[10.8, -3.3],[12.2, 11.8],[1.1, 4.9],[1.05, 3.95]])

    y_pred = softmax(y_logits)
    print(y_pred)

    loss = cross_entropy(y_true,y_pred)
    print(loss)