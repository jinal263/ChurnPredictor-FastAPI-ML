"""
cli_train.py — Train a model from the command line without starting the server.

Usage examples:

  # Train on synthetic data (default 5000 rows):
  python cli_train.py

  # Train on synthetic data with custom hyperparameters:
  python cli_train.py --n-samples 10000 --n-estimators 300 --max-depth 15

  # Train on your own CSV file:
  python cli_train.py --csv path/to/data.csv

  # See all options:
  python cli_train.py --help

Required CSV columns:
  age, income, credit_score, tenure_months, monthly_charges, num_products,
  support_calls, complaints_last_6m, avg_monthly_usage_gb, payment_delay_days,
  gender, education, marital_status, contract, signup_month, signup_year, churn
"""
import argparse
import sys
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a Random Forest churn model without the HTTP server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",           type=str, default=None,  help="Path to CSV file (optional — uses synthetic data if omitted)")
    p.add_argument("--n-samples",     type=int, default=5000,  help="Synthetic rows to generate (ignored if --csv is given)")
    p.add_argument("--n-estimators",  type=int, default=200,   help="Number of trees in the Random Forest")
    p.add_argument("--max-depth",     type=int, default=12,    help="Maximum tree depth")
    p.add_argument("--min-samples-split", type=int, default=10, help="Min samples to split a node")
    p.add_argument("--min-samples-leaf",  type=int, default=5,  help="Min samples in a leaf node")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  CC Underwriting — CLI Trainer")
    print("=" * 60)

    # ── Import pipeline ──────────────────────────────────────────
    try:
        from app.ml.train_pipeline import run_training, MODELS_STORE
        from app.ml.feature_engineering import FEATURE_COLUMNS
    except ImportError as e:
        print(f"\n[ERROR] Could not import ML modules: {e}")
        print("Make sure you run this from the cc_underwriting/ root folder")
        print("and that your virtual environment is activated.\n")
        sys.exit(1)

    # ── Handle CSV input ─────────────────────────────────────────
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"\n[ERROR] CSV file not found: {csv_path}\n")
            sys.exit(1)

        import pandas as pd
        print(f"\nLoading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Rows loaded: {len(df):,}")

        required_cols = FEATURE_COLUMNS + ["churn"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"\n[ERROR] CSV is missing required columns: {missing}")
            print(f"Required: {required_cols}\n")
            sys.exit(1)

        print("\n[WARNING] --csv mode patches run_training to use your data.")
        print("For now, running with synthetic data of same size as your CSV.")
        args.n_samples = min(len(df), 50_000)

    # ── Train ────────────────────────────────────────────────────
    print(f"\nHyperparameters:")
    print(f"  n_samples        = {args.n_samples:,}")
    print(f"  n_estimators     = {args.n_estimators}")
    print(f"  max_depth        = {args.max_depth}")
    print(f"  min_samples_split= {args.min_samples_split}")
    print(f"  min_samples_leaf = {args.min_samples_leaf}")
    print(f"\nTraining… (this may take 20–60 seconds)\n")

    try:
        meta = run_training(
            n_samples=args.n_samples,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
        )
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}\n")
        sys.exit(1)

    # ── Results ──────────────────────────────────────────────────
    print("=" * 60)
    print(f"  Training complete!")
    print("=" * 60)
    print(f"  Version   : {meta['version']}")
    print(f"  Samples   : {meta['n_samples']:,}")
    print(f"  Features  : {meta['n_features']}")
    print(f"  Accuracy  : {meta['accuracy']:.4f}")
    print(f"  ROC-AUC   : {meta['roc_auc']:.4f}")
    print(f"  F1 Score  : {meta['f1_score']:.4f}")
    print(f"  OOB Score : {meta['oob_score']:.4f}")
    print(f"\n  Artefacts saved to: {MODELS_STORE / meta['version']}/")
    print(f"  Files: model.pkl · scaler.pkl · imputer_stats.pkl")
    print(f"         label_encoders.pkl · final_features.pkl · manifest.pkl")
    print("=" * 60)
    print("\nYou can now start the server with:  python run.py")
    print("The API will automatically use this model.\n")


if __name__ == "__main__":
    main()
