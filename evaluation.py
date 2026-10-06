def evaluate(y_true, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # Treat >50K as the positive class
    for actual, predicted in zip(y_true, y_pred):
        if actual == ">50K" and predicted == ">50K":
            tp += 1
        elif actual == "<=50K" and predicted == "<=50K":
            tn += 1
        elif actual == "<=50K" and predicted == ">50K":
            fp += 1
        elif actual == ">50K" and predicted == "<=50K":
            fn += 1

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }