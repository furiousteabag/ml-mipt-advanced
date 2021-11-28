from collections import OrderedDict, Counter
from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np


class BoW(TransformerMixin):
    """
    Bag of words tranformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k

        # list of k most frequent tokens
        self.bow = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow
        counter = Counter()
        for text in X:
            counter.update(text.split())

        self.bow = [token for token, _ in counter.most_common()[:self.k]]

        # fit method must always return self
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """

        result = [0] * self.k
        for token in text.split():
            if token in self.bow:
                result[self.bow.index(token)] += 1

        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow


class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize

        # self.idf[term] = log(total # of documents / # of documents with term in it)
        self.idf = {}

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        
        for text in X:
            for token in text.split():
                if token in self.idf:
                    self.idf[token] += 1
                else:
                    self.idf[token] = 1

        self.idf = {k: v for k, v in sorted(self.idf.items(), key=lambda item: -item[1])}

        if self.k is not None:
            self.idf = {k: np.log(len(X) / v) for k, v in list(self.idf.items())[:self.k]}
        else:
            self.k = len(self.idf)
            self.idf = {k: np.log(len(X) / v) for k, v in self.idf.items()}

        # fit method must always return self
        return self

    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """
        result = [0] * self.k

        for token, n in Counter(text.split()).items():
            if token in self.idf:
                result[list(self.idf.keys()).index(token)] = n * self.idf[token] 

        result = np.array(result, "float32")

        if self.normalize == True:
            result = result / (np.linalg.norm(result) + 1e-6)
        
        return result

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])
