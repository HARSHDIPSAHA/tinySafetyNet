import pandas as pd
import mlflow
import os

def check_drift():
    # Load files
    tess = pd.read_csv("tinysafety-ops/data/tess_emotion_log.xlsx - Sheet1.csv")
    synth = pd.read_csv("tinysafety-ops/data/synthetic_emotion_inference.xlsx - Sheet1.csv")

    # Metric: Percentage of 'Fear' (TESS is known for Fear audio)
    tess_fear = (tess['inference_of_emotion'].str.lower() == 'fear').mean()
    synth_fear = (synth['inference_of_emotion'].str.lower() == 'fear').mean()

    drift_val = abs(tess_fear - synth_fear)

    # MLflow Tracking
    with mlflow.start_run(run_name="TinySafety_Drift_Audit"):
        mlflow.log_metric("fear_drift_magnitude", drift_val)
        mlflow.log_param("baseline_count", len(tess))
        mlflow.log_param("inference_count", len(synth))
        
        print(f"Fear Drift: {drift_val:.2%}")
        if drift_val > 0.20:
            print("ALERT: High Data Drift! Real-world data differs from training.")
            mlflow.set_tag("status", "DRIFT_DETECTED")
        else:
            print("Status: Stable")
            mlflow.set_tag("status", "HEALTHY")

if __name__ == "__main__":
    check_drift()