import numpy as np
import pandas as pd
from pandas import Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from utils import describe_as_df


class ElementwiseSummaryStats(BaseEstimator, TransformerMixin):
    """Constructs a transformer which computes the summary stats of each element of the input
    and concatenates the results into a single DataFrame.
    """

    def __init__(self, desc_kw_args):
        self.desc_kw_args = desc_kw_args

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.desc_kw_args is None:
            self.desc_kw_args = {}

        res = []

        for x_i in X:
            res.append(describe_as_df(x_i, desc_kw_args=self.desc_kw_args))

        output_df = pd.concat(res)

        if isinstance(X, Series):
            output_df.index = X.index
        else:
            output_df.reset_index(drop=True, inplace=True)

        return output_df


class ElementwiseTransformer(FunctionTransformer):
    """Constructs a transformer which applies a given function to each element of the input.
    Expects the data to be an iterable of elements, where each element is a single sample.
    Expects the output be of a consistent shape for each element.
    """

    def _transform(self, X, func=None, kw_args=None):
        # Construct a vectorized version of the function
        vfunc = np.vectorize(lambda x, **kwargs: func(x, **kwargs), otypes=[np.ndarray])

        # Apply the vectorized function to the input
        res = super()._transform(X, func=vfunc, kw_args=kw_args)

        if isinstance(X, Series):
            res = Series(res)
            res.index = X.index

        return res


class LibrosaTransformer(ElementwiseTransformer):
    """Constructs a transformer which applies a librosa function to each element of the input."""

    def _transform(self, X, func=None, kw_args=None):

        # Construct a new function which expects the audio data to be the first keyword argument
        def wrapped_func(x, **kwargs):
            return func(y=x, **kwargs)

        # Apply the vectorized function to the input
        return super()._transform(X, func=wrapped_func, kw_args=kw_args)
