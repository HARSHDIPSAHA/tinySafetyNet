import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd
import great_expectations as gx
from great_expectations.data_context.data_context import EphemeralDataContext
from great_expectations.core.expectation_suite import ExpectationSuite


REQUIRED_COLS = ["id", "time", "inference_of_emotion"]
VALID_LABELS = ["happy", "sad", "fear", "neutral", "angry", "disgust", "surprise"]
LABEL_NORMALIZE_MAP = {"surprised": "surprise"}


def read_csv_robust(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path, sep=None, engine="python", encoding="utf-8")
    except Exception:
        return pd.read_csv(file_path, sep=None, engine="python", encoding="iso-8859-1")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "inference_of_emotion" in df.columns:
        df["inference_of_emotion"] = (
            df["inference_of_emotion"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(LABEL_NORMALIZE_MAP)
        )
    return df


def get_context_filebacked():
    context = gx.get_context()
    if isinstance(context, EphemeralDataContext):
        context = context.convert_to_file_context()
    return context


def get_validator(context, df: pd.DataFrame, suite_name: str):
    ds_name = "pipeline_datasource"
    asset_name = "current_batch"

    try:
        datasource = context.data_sources.get(ds_name)
    except Exception:
        datasource = context.data_sources.add_pandas(name=ds_name)

    try:
        asset = datasource.get_asset(asset_name)
    except Exception:
        asset = datasource.add_dataframe_asset(name=asset_name)

    batch_request = asset.build_batch_request(options={"dataframe": df})

    try:
        context.suites.get(suite_name)
    except Exception:
        context.suites.add(ExpectationSuite(name=suite_name))

    return context.get_validator(batch_request=batch_request, expectation_suite_name=suite_name)


def diagnostics(df: pd.DataFrame) -> dict:
    info = {"row_count": int(len(df)), "columns": list(df.columns)}

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLS]
    info["missing_columns"] = missing
    info["extra_columns"] = extra

    if "inference_of_emotion" in df.columns:
        invalid_mask = ~df["inference_of_emotion"].isin(VALID_LABELS)
        invalid_vals = df.loc[invalid_mask, "inference_of_emotion"].tolist()
        info["invalid_label_count"] = int(invalid_mask.sum())
        info["top_invalid_labels"] = Counter(invalid_vals).most_common(20)
        info["sample_invalid_rows"] = (
            df.loc[invalid_mask, REQUIRED_COLS].head(15).to_dict(orient="records")
            if all(c in df.columns for c in REQUIRED_COLS)
            else []
        )

    # time parseability (pandas-side check; GX datetime expectations are version-sensitive)
    if "time" in df.columns:
        parsed = pd.to_datetime(df["time"], errors="coerce")
        info["time_unparseable_count"] = int(parsed.isna().sum())

    # id duplicates
    if "id" in df.columns:
        info["duplicate_id_count"] = int(df["id"].duplicated().sum())

    # null counts
    for c in REQUIRED_COLS:
        if c in df.columns:
            info[f"null_{c}_count"] = int(df[c].isna().sum())

    return info


def run_validation(file_path: str, suite_name: str, allow_extra_columns: bool, strict: bool) -> dict:
    print(f"--- GX Validating: {file_path} ---")
    if not os.path.exists(file_path):
        msg = f"CRITICAL ERROR: File {file_path} not found!"
        print(msg)
        return {"file": file_path, "success": False, "error": msg}

    df = normalize(read_csv_robust(file_path))
    ctx = get_context_filebacked()
    validator = get_validator(ctx, df, suite_name=suite_name)

    results = {}

    # 1) Not empty
    res_nonempty = validator.expect_table_row_count_to_be_between(min_value=1, max_value=None)
    results["row_count_nonzero"] = res_nonempty.success

    # 2) Columns / schema
    if allow_extra_columns:
        res_cols = validator.expect_table_columns_to_contain_set(column_set=REQUIRED_COLS)
    else:
        res_cols = validator.expect_table_columns_to_match_set(column_set=REQUIRED_COLS)
    results["schema_ok"] = res_cols.success

    # 3) Null checks
    for c in REQUIRED_COLS:
        if c in df.columns:
            res_null = validator.expect_column_values_to_not_be_null(c)
            results[f"not_null_{c}"] = res_null.success

    # 4) id uniqueness
    if "id" in df.columns:
        res_unique = validator.expect_column_values_to_be_unique("id")
        results["id_unique"] = res_unique.success

    # 5) label set check (after normalization)
    if "inference_of_emotion" in df.columns:
        res_labels = validator.expect_column_values_to_be_in_set("inference_of_emotion", VALID_LABELS)
        results["labels_valid"] = res_labels.success
    else:
        results["labels_valid"] = False

    # Final
    ok = all(results.values()) if strict else (results.get("schema_ok", False) and results.get("labels_valid", False))
    if ok:
        print(f"PASSED: {file_path} is healthy.\n")
    else:
        print(f"FAILED: Data validation failed for {file_path}")

    return {
        "file": file_path,
        "success": bool(ok),
        "checks": results,
        "diagnostics": diagnostics(df),
    }


def main():
    parser = argparse.ArgumentParser(description="TinySafetyNet: Great Expectations data validator")
    parser.add_argument("--tess", default="week5/data/tess_emotion_log.csv")
    parser.add_argument("--synth", default="week5/data/synthetic_emotion_inference.csv")
    parser.add_argument("--suite", default="integrity_suite")
    parser.add_argument("--out", default="validation_report.json")
    parser.add_argument("--allow-extra-columns", action="store_true")
    parser.add_argument("--non-strict", action="store_true", help="Only require schema+labels (skip uniqueness/null strictness)")
    args = parser.parse_args()

    strict = not args.non_strict

    report = {
        "suite": args.suite,
        "strict": strict,
        "allow_extra_columns": bool(args.allow_extra_columns),
        "runs": [],
        "overall_success": True,
    }

    for path in [args.tess, args.synth]:
        r = run_validation(path, suite_name=args.suite, allow_extra_columns=args.allow_extra_columns, strict=strict)
        report["runs"].append(r)
        if not r["success"]:
            report["overall_success"] = False

            # Helpful console diagnostics
            diag = r["diagnostics"]
            if diag.get("missing_columns"):
                print("Missing columns:", diag["missing_columns"])
            if diag.get("top_invalid_labels"):
                print("Top invalid labels:", diag["top_invalid_labels"])
            if diag.get("time_unparseable_count", 0) > 0:
                print("Unparseable time rows:", diag["time_unparseable_count"])
            if diag.get("duplicate_id_count", 0) > 0:
                print("Duplicate id rows:", diag["duplicate_id_count"])
            if diag.get("sample_invalid_rows"):
                print("Sample invalid rows:", diag["sample_invalid_rows"][:5])

    # Write artifact
    if args.out and args.out.strip():
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote report: {args.out}")

    sys.exit(0 if report["overall_success"] else 1)


if __name__ == "__main__":
    main()