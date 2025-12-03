import os
import pandas as pd

from preprocessing import EnhancedPreprocessor


def main():
    input_path = "data/processed/full_dataset.csv"
    if not os.path.exists(input_path):
        print(f"full_dataset.csv not found at {input_path}. Run preprocessing.py first.")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded full_dataset with {len(df)} rows.")

    # Use the same splitting logic as EnhancedPreprocessor.split_dataset
    preprocessor = EnhancedPreprocessor()
    train_df, val_df, test_df = preprocessor.split_dataset(df)

    out_dir = preprocessor.output_dir
    train_path = out_dir / "full_dataset_train.csv"
    val_path = out_dir / "full_dataset_val.csv"
    test_path = out_dir / "full_dataset_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved: {train_path} ({len(train_df)})")
    print(f"Saved: {val_path} ({len(val_df)})")
    print(f"Saved: {test_path} ({len(test_df)})")


if __name__ == "__main__":
    main()
