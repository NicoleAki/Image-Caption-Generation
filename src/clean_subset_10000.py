import os
import pandas as pd


def main():
    input_path = 'data/processed/subset_10000.csv'
    output_path = 'data/processed/subset_10000_clean.csv'

    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    if 'full_path' not in df.columns:
        print("Column 'full_path' not found in CSV. Nothing to clean.")
        print(f"Columns present: {list(df.columns)}")
        return

    total = len(df)
    print(f"Loaded {total} rows from {input_path}")

    # Keep only rows where the image file actually exists
    exists_mask = df['full_path'].apply(os.path.exists)
    cleaned = df[exists_mask].copy()

    kept = len(cleaned)
    dropped = total - kept
    print(f"Keeping {kept} rows with existing images; dropping {dropped} rows with missing images.")

    cleaned.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to {output_path}")


if __name__ == '__main__':
    main()
