from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing._function_transformer import _identity


class ElementwiseTransformer(FunctionTransformer):
    """Constructs a transformer which applies a given function to each element of the input.
    Expects the data to be an iterable of elements, where each element is a single sample.
    Expects the output be of a consistent shape for each element.
    """

    def _transform(self, X, func=None, kw_args=None):
        if func is None:
            func = _identity

        out = []

        for x_i in X:
            out.append(func(x_i, **(kw_args if kw_args else {})))

        return out
