"""
Create Metadata from WikiArt Dataset
Handles nested folder structure (style/artist folders)
Maps images to ArtEmis captions
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import pickle
from PIL import Image
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def scan_wikiart_images(wikiart_dir):
    """
    Scan WikiArt directory and create image inventory
    Handles nested folder structure
    
    Args:
        wikiart_dir: Path to WikiArt root directory
        
    Returns:
        DataFrame with columns: [full_path, relative_path, style, filename, valid]
    """
    logger.info(f"Scanning WikiArt directory: {wikiart_dir}")
    
    wikiart_path = Path(wikiart_dir)
    image_records = []
    
    # Walk through all subdirectories
    for style_folder in tqdm(list(wikiart_path.iterdir()), desc="Scanning styles"):
        if not style_folder.is_dir():
            continue
            
        style_name = style_folder.name
        
        # Get all images in this style folder (including subfolders)
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
        for img_path in style_folder.rglob('*'):
            if img_path.suffix in image_extensions:
                # Check if image is valid
                is_valid = validate_image(img_path)
                
                if is_valid:
                    record = {
                        'full_path': str(img_path),
                        'relative_path': str(img_path.relative_to(wikiart_path)),
                        'style': style_name,
                        'filename': img_path.name,
                        'valid': is_valid
                    }
                    image_records.append(record)
    
    df = pd.DataFrame(image_records)
    logger.info(f"Found {len(df)} valid images across {df['style'].nunique()} styles")
    
    # Show style distribution
    logger.info("\nImage distribution by style:")
    style_counts = df['style'].value_counts()
    for style, count in style_counts.head(10).items():
        logger.info(f"  {style}: {count}")
    
    return df


def validate_image(img_path):
    """
    Validate that image file is not corrupted
    
    Args:
        img_path: Path to image file
        
    Returns:
        Boolean indicating if image is valid
    """
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.warning(f"Corrupt image: {img_path}")
        return False


def load_artemis_captions(captions_file):
    """
    Load ArtEmis captions
    Handles both CSV and JSON formats
    
    Args:
        captions_file: Path to captions file
        
    Returns:
        DataFrame with captions
    """
    logger.info(f"Loading ArtEmis captions from: {captions_file}")
    
    if captions_file.endswith('.csv'):
        df = pd.read_csv(captions_file)
    elif captions_file.endswith('.json'):
        with open(captions_file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError("Captions file must be CSV or JSON")
    
    logger.info(f"Loaded {len(df)} caption entries")
    
    # Show column names
    logger.info(f"Caption columns: {df.columns.tolist()}")
    
    return df


def match_images_to_captions(images_df, captions_df):
    """
    Match WikiArt images to ArtEmis captions
    
    Args:
        images_df: DataFrame of scanned images
        captions_df: DataFrame of ArtEmis captions
        
    Returns:
        Merged DataFrame with matched images and captions
    """
    logger.info("Matching images to captions...")
    
    # ArtEmis captions might have different column names
    # Common possibilities: 'painting', 'image_file', 'art_style', 'style'
    
    # Detect caption column names
    caption_col = None
    for col in ['utterance', 'caption', 'text', 'description','emotion']:
        if col in captions_df.columns:
            caption_col = col
            break
    
    if caption_col is None:
        raise ValueError(f"Could not find caption column. Available: {captions_df.columns.tolist()}")
    
    # Detect image filename column
    filename_col = None
    for col in ['painting', 'image_file', 'filename', 'art_path','art_style']:
        if col in captions_df.columns:
            filename_col = col
            break
    
    if filename_col is None:
        raise ValueError(f"Could not find filename column. Available: {captions_df.columns.tolist()}")
    
    logger.info(f"Using caption column: '{caption_col}'")
    logger.info(f"Using filename column: '{filename_col}'")
    # Clean caption filenames (ArtEmis does NOT include file extensions)
    def normalize_filename(name):
        if pd.isna(name):
            return None
        base = Path(name).stem  # remove extension if it exists
        return base.lower()
    
    # Create matching key - extract just the filename from paths
    images_df['match_key'] = images_df['filename'].apply(lambda x: Path(x).stem.lower())
    
    # Clean caption filenames (might have full paths or just filenames)
    captions_df['match_key'] = captions_df[filename_col].apply(
        normalize_filename)
    
    # Merge
    merged_df = images_df.merge(
        captions_df,
        on='match_key',
        how='inner'
    )
    
    logger.info(f"Matched {len(merged_df)} images with captions")
    logger.info(f"Unmatched images: {len(images_df) - len(merged_df)}")
    
    # Rename caption column to standard name
    merged_df = merged_df.rename(columns={caption_col: 'caption'})
    
    # Select and order columns
    final_columns = ['full_path', 'relative_path', 'style', 'filename', 'caption']
    
    # Add any additional columns that might be useful
    if 'emotion' in merged_df.columns:
        final_columns.append('emotion')
    if 'artist' in merged_df.columns:
        final_columns.append('artist')
    
    # Keep only columns that exist
    final_columns = [col for col in final_columns if col in merged_df.columns]
    merged_df = merged_df[final_columns]
    
    return merged_df


def create_stratified_subsets(df, sizes=[1000, 2000, 5000, 10000], stratify_by='style'):
    """
    Create stratified subsets preserving style distribution
    
    Args:
        df: Full dataset DataFrame
        sizes: List of subset sizes to create
        stratify_by: Column to stratify by
        
    Returns:
        Dictionary of subset DataFrames
    """
    logger.info(f"Creating stratified subsets by '{stratify_by}'...")
    
    subsets = {}
    
    for size in sizes:
        if size > len(df):
            logger.warning(f"Subset size {size} larger than dataset ({len(df)}), skipping")
            continue
        
        # Calculate samples per stratum
        strata = df[stratify_by].value_counts()
        sampled_dfs = []
        
        for stratum, count in strata.items():
            # Proportional sampling
            stratum_size = int(size * count / len(df))
            
            if stratum_size > 0:
                stratum_df = df[df[stratify_by] == stratum]
                sample_size = min(stratum_size, len(stratum_df))
                
                sampled = stratum_df.sample(n=sample_size, random_state=42)
                sampled_dfs.append(sampled)
        
        subset_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Shuffle
        subset_df = subset_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        subsets[f'subset_{size}'] = subset_df
        logger.info(f"  Created subset_{size}: {len(subset_df)} samples")
    
    return subsets


def build_vocabulary(captions, vocab_size=10000, min_freq=2):
    """
    Build vocabulary from captions
    
    Args:
        captions: List of caption strings
        vocab_size: Maximum vocabulary size
        min_freq: Minimum word frequency
        
    Returns:
        Dictionary with word2idx, idx2word, and vocab list
    """
    logger.info("Building vocabulary...")
    
    from collections import Counter
    import re
    
    # Tokenize and count words
    word_counts = Counter()
    
    for caption in tqdm(captions, desc="Processing captions"):
        # Clean and tokenize
        caption = caption.lower()
        caption = re.sub(r'[^a-z0-9\s]', '', caption)
        words = caption.split()
        word_counts.update(words)
    
    logger.info(f"Found {len(word_counts)} unique words")
    
    # Filter by frequency and select top words
    filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
    top_words = sorted(filtered_words, key=lambda x: word_counts[x], reverse=True)[:vocab_size-4]
    
    # Build vocabulary with special tokens
    vocab = ['<PAD>', '<START>', '<END>', '<UNK>'] + top_words
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    
    logger.info(f"Vocabulary size: {len(vocab)}")
    logger.info(f"Coverage: {len(top_words) / len(word_counts) * 100:.2f}%")
    
    vocab_data = {
        'word2idx': word2idx,
        'idx2word': idx2word,
        'vocab': vocab,
        'vocab_size': len(vocab)
    }
    
    return vocab_data


def save_metadata(df, output_path):
    """Save metadata DataFrame"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved metadata to {output_path}")


def main():
    """Main function to create all metadata"""
    
    # Configuration
    WIKIART_DIR = 'data/raw/wikiart'
    CAPTIONS_FILE = 'data/raw/artemis_captions.csv'  # Adjust filename
    OUTPUT_DIR = Path('data/processed')
    
    logger.info("="*60)
    logger.info("CREATING METADATA FOR WIKIART + ARTEMIS")
    logger.info("="*60)
    
    # Step 1: Scan WikiArt images
    logger.info("\n[1/5] Scanning WikiArt images...")
    images_df = scan_wikiart_images(WIKIART_DIR)
    
    # Save full image inventory
    images_df.to_csv(OUTPUT_DIR / 'full_image_inventory.csv', index=False)
    logger.info(f"Saved full inventory: {len(images_df)} images")
    
    # Step 2: Load ArtEmis captions
    logger.info("\n[2/5] Loading ArtEmis captions...")
    captions_df = load_artemis_captions(CAPTIONS_FILE)
    
    # Step 3: Match images to captions
    logger.info("\n[3/5] Matching images to captions...")
    matched_df = match_images_to_captions(images_df, captions_df)
    
    # Save full matched dataset
    matched_df.to_csv(OUTPUT_DIR / 'full_dataset.csv', index=False)
    logger.info(f"Saved full dataset: {len(matched_df)} samples")
    
    # Step 4: Create stratified subsets
    logger.info("\n[4/5] Creating stratified subsets...")
    subsets = create_stratified_subsets(matched_df, sizes=[1000, 2000, 5000, 10000])
    
    for name, subset_df in subsets.items():
        save_metadata(subset_df, OUTPUT_DIR / f'{name}.csv')
    
    # Step 5: Build vocabulary
    logger.info("\n[5/5] Building vocabulary...")
    vocab_data = build_vocabulary(matched_df['caption'].tolist())
    
    with open(OUTPUT_DIR / 'vocabulary.pkl', 'wb') as f:
        pickle.dump(vocab_data, f)
    logger.info(f"Saved vocabulary to {OUTPUT_DIR / 'vocabulary.pkl'}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("METADATA CREATION COMPLETE!")
    logger.info("="*60)
    logger.info(f"\nCreated files:")
    logger.info(f"  ✓ full_image_inventory.csv - All {len(images_df)} images")
    logger.info(f"  ✓ full_dataset.csv - {len(matched_df)} matched images+captions")
    for name, subset_df in subsets.items():
        logger.info(f"  ✓ {name}.csv - {len(subset_df)} samples")
    logger.info(f"  ✓ vocabulary.pkl - {vocab_data['vocab_size']} words")
    logger.info(f"\nNext step: python src/train.py --subset_size 1000")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create metadata from WikiArt + ArtEmis')
    parser.add_argument('--wikiart_dir', type=str, default='data/raw/wikiart',
                       help='Path to WikiArt directory')
    parser.add_argument('--captions_file', type=str, default='data/raw/artemis_captions.csv',
                       help='Path to ArtEmis captions file')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for metadata')
    
    args = parser.parse_args()
    
    # Update globals
    WIKIART_DIR = args.wikiart_dir
    CAPTIONS_FILE = args.captions_file
    OUTPUT_DIR = Path(args.output_dir)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run
    main()