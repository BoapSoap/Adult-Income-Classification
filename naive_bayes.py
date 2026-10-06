import math
import numpy as np


NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

TARGET = "income"


# Middle bin widths from the EDA comparisons
BIN_WIDTHS = {
    "age": 10,
    "fnlwgt": 50000,
    "education-num": 2,
    "capital-gain": 5000,
    "capital-loss": 500,
    "hours-per-week": 10
}


class NaiveBayes:
    def __init__(self, mode="discretized"):
        if mode not in ["discretized", "gaussian"]:
            raise ValueError("Mode must be 'discretized' or 'gaussian'")

        self.mode = mode
        self.classes = []

        self.class_counts = {}
        self.class_priors = {}

        self.value_counts = {}
        self.possible_values = {}

        self.gaussian_stats = {}


    def discretize(self, value, width):
        return math.floor(value / width) * width


    def fit(self, data):
        self.classes = list(data[TARGET].unique())
        total = len(data)

        # Calculate the prior probability for each income class
        for c in self.classes:
            class_data = data[data[TARGET] == c]

            self.class_counts[c] = len(class_data)
            self.class_priors[c] = len(class_data) / total

        # Store counts for each categorical attribute by class
        for column in CATEGORICAL_COLUMNS:
            self.possible_values[column] = list(data[column].unique())
            self.value_counts[column] = {}

            for c in self.classes:
                class_data = data[data[TARGET] == c]

                self.value_counts[column][c] = (
                    class_data[column].value_counts().to_dict()
                )

        if self.mode == "discretized":
            self.fit_discretized(data)

        else:
            self.fit_gaussian(data)


    def fit_discretized(self, data):
        # Numeric attributes are converted into equal-width bins
        for column in NUMERIC_COLUMNS:
            width = BIN_WIDTHS[column]

            binned = data[column].apply(
                lambda value: self.discretize(value, width)
            )

            self.possible_values[column] = list(binned.unique())
            self.value_counts[column] = {}

            for c in self.classes:
                mask = data[TARGET] == c

                self.value_counts[column][c] = (
                    binned[mask].value_counts().to_dict()
                )


    def fit_gaussian(self, data):
        # Keep numeric attributes continuous and learn their distribution
        for column in NUMERIC_COLUMNS:
            self.gaussian_stats[column] = {}

            for c in self.classes:
                values = data.loc[data[TARGET] == c, column]

                self.gaussian_stats[column][c] = {
                    "mean": values.mean(),
                    "variance": values.var(ddof=0)
                }


    def categorical_log_prob(self, column, value, c):
        counts = self.value_counts[column][c]

        count = counts.get(value, 0)
        class_count = self.class_counts[c]
        num_values = len(self.possible_values[column])

        # Laplace correction prevents an unseen value from giving probability 0
        probability = (count + 1) / (class_count + num_values)

        return math.log(probability)


    def gaussian_log_prob(self, column, value, c):
        mean = self.gaussian_stats[column][c]["mean"]
        variance = self.gaussian_stats[column][c]["variance"]

        # Prevent division by zero if a feature has no variance
        if variance == 0:
            variance = 1e-9

        return (
            -0.5 * math.log(2 * math.pi * variance)
            - ((value - mean) ** 2) / (2 * variance)
        )


    def predict_one(self, row):
        scores = {}

        for c in self.classes:
            # Start with the prior probability of the class
            score = math.log(self.class_priors[c])

            for column in CATEGORICAL_COLUMNS:
                score += self.categorical_log_prob(
                    column,
                    row[column],
                    c
                )

            for column in NUMERIC_COLUMNS:
                value = row[column]

                if self.mode == "discretized":
                    value = self.discretize(
                        value,
                        BIN_WIDTHS[column]
                    )

                    score += self.categorical_log_prob(
                        column,
                        value,
                        c
                    )

                else:
                    score += self.gaussian_log_prob(
                        column,
                        value,
                        c
                    )

            scores[c] = score

        return max(scores, key=scores.get)


    def predict(self, data):
        predictions = []

        for _, row in data.iterrows():
            predictions.append(self.predict_one(row))

        return np.array(predictions)