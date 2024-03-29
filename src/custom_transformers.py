from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing._function_transformer import _identity
import numpy as np
from pandas import Series


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
