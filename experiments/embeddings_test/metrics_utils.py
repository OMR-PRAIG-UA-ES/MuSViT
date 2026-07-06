"""Metric helpers used by the training/evaluation scripts."""

from sklearn.metrics import accuracy_score, f1_score


def accuracy(predictions, expected):
    return accuracy_score(expected, predictions)


def macro_fscore(predictions, expected):
    return f1_score(expected, predictions, average="macro")


def micro_fscore(predictions, expected):
    return f1_score(expected, predictions, average="micro")


def weighted_fscore(predictions, expected):
    return f1_score(expected, predictions, average="weighted")


# Backwards-compatible aliases used by the original script.
Accuracy = accuracy
MacroFscore = macro_fscore
MicroFscore = micro_fscore
WeightedFscore = weighted_fscore


def is_better(config, result_val, result_val_max):
    if config.validation_monitor == "max":
        return result_val > result_val_max
    return result_val < result_val_max


# Backwards-compatible alias.
isBetter = is_better
