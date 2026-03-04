import pandas as pd
import great_expectations as ge
import sys
import os

def run_validation(file_path):
    print(f"--- Great Expectations Validating: {file_path} ---")
    if not os.path.exists(file_path):
        print(f"CRITICAL ERROR: File {file_path} not found!")
        return False

    # Load data into a Great Expectations Dataset
    df = ge.from_pandas(pd.read_csv(file_path))

    # 1. Expectation: Required columns must exist
    # This replaces your manual column list check
    res_cols = df.expect_table_columns_to_match_ordered_list(
        column_list=['id', 'time', 'inference_of_emotion']
    )

    # 2. Expectation: Emotions must be in the valid set
    # This catches typos like "Feear" or "happyy"
    valid_labels = ['happy', 'sad', 'fear', 'neutral', 'angry', 'disgust']
    res_labels = df.expect_column_values_to_be_in_set(
        "inference_of_emotion", 
        valid_labels,
        ignore_row_if="none"
    )

    # 3. Expectation: 'id' column must not have null values
    res_nulls = df.expect_column_values_to_not_be_null("id")

    # Final check: Did all expectations pass?
    if not (res_cols.success and res_labels.success and res_nulls.success):
        print(f"FAILED: Data validation failed for {file_path}")
        # Print specific failures for debugging
        if not res_labels.success:
            print(f"Unexpected values found: {res_labels.result['unexpected_list']}")
        return False

    print(f"PASSED: {file_path} meets all expectations.\n")
    return True

if __name__ == "__main__":
    # Ensure the paths match your monorepo structure
    tess_file = "tinysafety-ops/data/tess_emotion_log.xlsx - Sheet1.csv"
    synth_file = "tinysafety-ops/data/synthetic_emotion_inference.xlsx - Sheet1.csv"

    if not (run_validation(tess_file) and run_validation(synth_file)):
        sys.exit(1)