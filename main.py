import pandas as pd

from naive_bayes import NaiveBayes
from evaluation import evaluate
from decision_tree import (
    cross_validate_decision_tree,
    test_decision_tree
)
from cross_validation import (
    cross_validate,
    get_imputation_values,
    apply_imputation
)

DATA_PATH = "adult.csv"

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

TARGET_COLUMN = "income"


def load_data(path):
    df = pd.read_csv(path, skipinitialspace=True)

    # Convert Adult's missing-value marker into a standard missing value.
    df = df.replace("?", pd.NA)

    return df


def handle_missing_drop(df):
    """Strategy 1: remove rows containing missing values."""
    return df.dropna().reset_index(drop=True)


def handle_missing_impute(df):
    """
    Strategy 2:
    - categorical attributes -> mode
    - numerical attributes -> median
    """
    df = df.copy()

    for column in CATEGORICAL_COLUMNS:
        if df[column].isna().any():
            mode = df[column].mode()[0]
            df[column] = df[column].fillna(mode)

    for column in NUMERIC_COLUMNS:
        if df[column].isna().any():
            median = df[column].median()
            df[column] = df[column].fillna(median)

    return df

def load_test_data(path):
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income"
    ]

    test_data = pd.read_csv(
        path,
        names=columns,
        skiprows=1,
        skipinitialspace=True
    )

    # Missing values use the same ? marker as the training data
    test_data = test_data.replace("?", pd.NA)

    # adult.test stores the income labels with a period at the end
    test_data["income"] = test_data["income"].str.rstrip(".")

    return test_data


if __name__ == "__main__":
    data = load_data(DATA_PATH)

    print("Original rows:", len(data))
    print("\nMissing values:")
    print(data.isna().sum())

    dropped_data = handle_missing_drop(data)
    imputed_data = handle_missing_impute(data)

    print("\nRows after deletion:", len(dropped_data))
    print("Rows after imputation:", len(imputed_data))

    print("\nMissing after deletion:")
    print(dropped_data.isna().sum())

    print("\nMissing after imputation:")
    print(imputed_data.isna().sum())

    print("\n===== NAIVE BAYES 10-FOLD CROSS VALIDATION =====")

    experiments = [
        ("discretized", "drop"),
        ("discretized", "impute"),
        ("gaussian", "drop"),
        ("gaussian", "impute")
    ]

    for mode, missing_strategy in experiments:
        print(
            f"\n--- {mode.upper()} + {missing_strategy.upper()} ---"
        )

        results = cross_validate(
            data,
            mode=mode,
            missing_strategy=missing_strategy
        )

        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1 Score:  {results['f1']:.4f}")

        print(
            f"TP: {results['tp']}  "
            f"TN: {results['tn']}  "
            f"FP: {results['fp']}  "
            f"FN: {results['fn']}"
        )

    print("\n===== DECISION TREE 10-FOLD CROSS VALIDATION =====")

    tree_results = cross_validate_decision_tree(data)

    print(f"Accuracy:  {tree_results['accuracy']:.4f}")
    print(f"Precision: {tree_results['precision']:.4f}")
    print(f"Recall:    {tree_results['recall']:.4f}")
    print(f"F1 Score:  {tree_results['f1']:.4f}")

    print(
        f"TP: {tree_results['tp']}  "
        f"TN: {tree_results['tn']}  "
        f"FP: {tree_results['fp']}  "
        f"FN: {tree_results['fn']}"
    )

    print("\n===== FINAL TEST SET EVALUATION =====")
    print("\n--- NAIVE BAYES ---")
    # The final configuration is discretization with imputation
    test_data = load_test_data("adult.test")

    # Learn missing-value replacements from the training data only
    imputation_values = get_imputation_values(data)

    final_train_data = apply_imputation(
        data,
        imputation_values
    )

    final_test_data = apply_imputation(
        test_data,
        imputation_values
    )

    # Train one final model using all available training data
    final_model = NaiveBayes(mode="discretized")
    final_model.fit(final_train_data)

    final_predictions = final_model.predict(final_test_data)

    final_results = evaluate(
        final_test_data["income"].to_numpy(),
        final_predictions
    )

    print("Test rows:", len(final_test_data))
    print(f"Accuracy:  {final_results['accuracy']:.4f}")
    print(f"Precision: {final_results['precision']:.4f}")
    print(f"Recall:    {final_results['recall']:.4f}")
    print(f"F1 Score:  {final_results['f1']:.4f}")

    print(
        f"TP: {final_results['tp']}  "
        f"TN: {final_results['tn']}  "
        f"FP: {final_results['fp']}  "
        f"FN: {final_results['fn']}"
    )
    print("\n--- DECISION TREE ---")

    tree_test_results = test_decision_tree(
        data,
        test_data
    )

    print("Test rows:", len(test_data))
    print(f"Accuracy:  {tree_test_results['accuracy']:.4f}")
    print(f"Precision: {tree_test_results['precision']:.4f}")
    print(f"Recall:    {tree_test_results['recall']:.4f}")
    print(f"F1 Score:  {tree_test_results['f1']:.4f}")

    print(
        f"TP: {tree_test_results['tp']}  "
        f"TN: {tree_test_results['tn']}  "
        f"FP: {tree_test_results['fp']}  "
        f"FN: {tree_test_results['fn']}"
    )