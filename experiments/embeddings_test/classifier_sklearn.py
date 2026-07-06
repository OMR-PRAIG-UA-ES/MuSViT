"""Small callable wrapper around scikit-learn classifiers."""


class ClassifierSKL:
    """Expose a scikit-learn classifier through a simple callable interface."""

    def __init__(self, classifier_model):
        self.classifier_model = classifier_model

    def __call__(self, *inputs):
        if len(inputs) == 1:
            return self._call_predict(inputs[0])
        if len(inputs) in (2, 3):
            # The third argument is ignored for compatibility with other training call signatures.
            return self._call_training(inputs[0], inputs[1])
        raise ValueError("Unsupported number of arguments")

    def _call_predict(self, x):
        return self.classifier_model.predict(x)

    def _call_training(self, x, y):
        return self.classifier_model.fit(x, y)
