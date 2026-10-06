import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

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


def prepare_data(train_data, validation_data):
    train_data = train_data.copy()
    validation_data = validation_data.copy()

    # Convert categorical attributes into numerical columns
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    train_categorical = encoder.fit_transform(
        train_data[CATEGORICAL_COLUMNS]
    )

    validation_categorical = encoder.transform(
        validation_data[CATEGORICAL_COLUMNS]
    )

    train_numeric = train_data[NUMERIC_COLUMNS].to_numpy()
    validation_numeric = validation_data[NUMERIC_COLUMNS].to_numpy()

    # Combine the numeric and encoded categorical attributes
    x_train = np.hstack(
        (train_numeric, train_categorical)
    )

    x_validation = np.hstack(
        (validation_numeric, validation_categorical)
    )

    y_train = train_data[TARGET].to_numpy()
    y_validation = validation_data[TARGET].to_numpy()

    return x_train, x_validation, y_train, y_validation


def cross_validate_decision_tree(data, k=10, seed=42):
    # Shuffle the data before creating the folds
    shuffled = data.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    folds = np.array_split(
        np.arange(len(shuffled)),
        k
    )

    all_actual = []
    all_predictions = []

    for i in range(k):
        validation_indices = folds[i]

        training_indices = np.concatenate(
            [folds[j] for j in range(k) if j != i]
        )

        train_data = shuffled.iloc[training_indices].copy()
        validation_data = shuffled.iloc[validation_indices].copy()

        # Learn missing-value replacements from the training folds only
        imputation_values = get_imputation_values(train_data)

        train_data = apply_imputation(
            train_data,
            imputation_values
        )

        validation_data = apply_imputation(
            validation_data,
            imputation_values
        )

        x_train, x_validation, y_train, y_validation = prepare_data(
            train_data,
            validation_data
        )

        model = DecisionTreeClassifier(
            random_state=seed
        )

        model.fit(x_train, y_train)

        predictions = model.predict(x_validation)

        all_actual.extend(y_validation)
        all_predictions.extend(predictions)

        print(f"Decision Tree fold {i + 1}/{k} complete")

    return evaluate(
        all_actual,
        all_predictions
    )

def test_decision_tree(train_data, test_data, seed=42):
    # Learn missing-value replacements from the training data only
    imputation_values = get_imputation_values(train_data)

    train_data = apply_imputation(
        train_data,
        imputation_values
    )

    test_data = apply_imputation(
        test_data,
        imputation_values
    )

    x_train, x_test, y_train, y_test = prepare_data(
        train_data,
        test_data
    )

    model = DecisionTreeClassifier(
        random_state=seed
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    return evaluate(
        y_test,
        predictions
    )