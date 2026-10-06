import numpy as np

from naive_bayes import NaiveBayes
from evaluation import evaluate


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


def get_imputation_values(train_data):
    # Missing-value replacements are learned only from the training folds
    values = {}

    for column in CATEGORICAL_COLUMNS:
        values[column] = train_data[column].mode()[0]

    for column in NUMERIC_COLUMNS:
        values[column] = train_data[column].median()

    return values


def apply_imputation(data, values):
    data = data.copy()

    for column in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS:
        data[column] = data[column].fillna(values[column])

    return data


def cross_validate(data, mode, missing_strategy, k=10, seed=42):
    # Shuffle before creating the folds
    shuffled = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split the row indices into k approximately equal folds
    folds = np.array_split(np.arange(len(shuffled)), k)

    all_actual = []
    all_predictions = []

    for i in range(k):
        validation_indices = folds[i]

        training_indices = np.concatenate(
            [folds[j] for j in range(k) if j != i]
        )

        train_data = shuffled.iloc[training_indices].copy()
        validation_data = shuffled.iloc[validation_indices].copy()

        if missing_strategy == "drop":
            train_data = train_data.dropna()
            validation_data = validation_data.dropna()

        elif missing_strategy == "impute":
            imputation_values = get_imputation_values(train_data)

            train_data = apply_imputation(
                train_data,
                imputation_values
            )

            validation_data = apply_imputation(
                validation_data,
                imputation_values
            )

        else:
            raise ValueError(
                "Missing strategy must be 'drop' or 'impute'"
            )

        model = NaiveBayes(mode=mode)
        model.fit(train_data)

        predictions = model.predict(validation_data)

        all_actual.extend(validation_data[TARGET].to_numpy())
        all_predictions.extend(predictions)

        print(f"Fold {i + 1}/{k} complete")

    return evaluate(all_actual, all_predictions)