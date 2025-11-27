"""
Unified Preprocessing Pipeline for WikiArt + ArtEmis Dataset
Combines metadata creation and preprocessing with all optimizations

Features:
- Image resolution reduction (224×224)
- Stratified subsampling by art style
- Progressive dataset scaling (1k, 5k, 10k)
- Vocabulary building with 10k words
- Caption length limiting (50 words max)
- Corrupt file detection
- Memory-efficient on-the-fly loading
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import pickle
from PIL import Image
import re
from collections import Counter
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedPreprocessor:
    """
    Complete preprocessing pipeline for WikiArt + ArtEmis
    """
    
    def __init__(self, 
                 wikiart_dir='data/raw/wikiart',
                 captions_file='data/raw/artemis_captions.csv',
                 output_dir='data/processed',
                 image_size=224,
                 vocab_size=10000,
                 max_caption_length=50,
                 subset_sizes=[1000, 5000, 10000]):
        
        self.wikiart_dir = Path(wikiart_dir)
        self.captions_file = captions_file
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.vocab_size = vocab_size
        self.max_caption_length = max_caption_length
        self.subset_sizes = subset_sizes
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_output_dir = self.output_dir / 'resized_images'
        self.images_output_dir.mkdir(exist_ok=True)

        logger.info("="*70)
        logger.info("UNIFIED PREPROCESSING PIPELINE")
        logger.info("="*70)
        logger.info(f"Configuration:")
        logger.info(f"  WikiArt directory: {self.wikiart_dir}")
        logger.info(f"  Captions file: {self.captions_file}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Image size: {self.image_size}×{self.image_size}")
        logger.info(f"  Vocabulary size: {self.vocab_size}")
        logger.info(f"  Max caption length: {self.max_caption_length} words")
        logger.info(f"  Subset sizes: {self.subset_sizes}")
    
    def validate_image_and_resize_image(self, img_path):
        """
        Validate that image file is not corrupted
        Resize images to image_size x image_size
        ️Returns True if valid, False if corrupt
        """
        try:
            with Image.open(img_path) as img:
                img.verify()
            return self.resize_image(img_path)
        except Exception as e:
            logger.warning(f"Corrupt image: {img_path}")
            return False
        

    def resize_image(self, img_path):
        """Resize image to target size and save to output directory"""
        try:
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img_resized = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
                
                # Save resized image
                relative_path = Path(img_path).relative_to(self.wikiart_dir)
                output_path = self.images_output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                img_resized.save(output_path, JPG quality=90)
                return True
        except Exception as e:
            logger.warning(f"Failed to resize image: {img_path} : {e}")
            return False
    
    def scan_wikiart_images(self):
        """
        Step 1: Scan WikiArt directory and validate images
        Instructor recommendation: Monitor and clean corrupt files
        """
        logger.info("\n[STEP 1/6] Scanning WikiArt images and validating...")
        
        image_records = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
        # Walk through all style subdirectories
        for style_folder in tqdm(list(self.wikiart_dir.iterdir()), 
                                desc="Scanning styles"):
            if not style_folder.is_dir():
                continue
            
            style_name = style_folder.name
            
            # Get all images in this style folder
            for img_path in style_folder.rglob('*'):
                if img_path.suffix in image_extensions:
                    # Validate image (corrupt file detection)
                    if self.validate_image(img_path):
                        record = {
                            'full_path': str(img_path),
                            'relative_path': str(img_path.relative_to(self.wikiart_dir)),
                            'style': style_name,
                            'filename': img_path.name
                        }
                        image_records.append(record)
        
        images_df = pd.DataFrame(image_records)
        logger.info(f"✓ Found {len(images_df):,} valid images across "
                   f"{images_df['style'].nunique()} styles")
        
        # Show style distribution
        logger.info("\n  Top 10 styles by count:")
        for style, count in images_df['style'].value_counts().head(10).items():
            logger.info(f"    {style}: {count:,}")
        
        return images_df
    
    def load_artemis_captions(self):
        """
        Step 2: Load ArtEmis captions
        """
        logger.info(f"\n[STEP 2/6] Loading ArtEmis captions...")
        
        if self.captions_file.endswith('.csv'):
            df = pd.read_csv(self.captions_file)
        elif self.captions_file.endswith('.json'):
            with open(self.captions_file, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError("Captions file must be CSV or JSON")
        
        logger.info(f"✓ Loaded {len(df):,} caption entries")
        logger.info(f"  Columns: {df.columns.tolist()}")
        
        return df
    
    def match_images_to_captions(self, images_df, captions_df):
        """
        Step 3: Match WikiArt images to ArtEmis captions
        """
        logger.info("\n[STEP 3/6] Matching images to captions...")
        
        # Detect caption column
        caption_col = None
        for col in ['utterance', 'caption', 'text', 'description']:
            if col in captions_df.columns:
                caption_col = col
                break
        
        if caption_col is None:
            raise ValueError(f"Caption column not found. Available: {captions_df.columns.tolist()}")
        
        # Detect image filename column
        filename_col = None
        for col in ['painting', 'image_file', 'filename', 'art_path', 'art_style']:
            if col in captions_df.columns:
                filename_col = col
                break
        
        if filename_col is None:
            raise ValueError(f"Filename column not found. Available: {captions_df.columns.tolist()}")
        
        logger.info(f"  Using caption column: '{caption_col}'")
        logger.info(f"  Using filename column: '{filename_col}'")
        
        # Create matching keys (normalize filenames)
        images_df['match_key'] = images_df['filename'].apply(
            lambda x: Path(x).stem.lower()
        )
        captions_df['match_key'] = captions_df[filename_col].apply(
            lambda x: Path(str(x)).stem.lower() if pd.notna(x) else None
        )
        
        # Merge
        merged_df = images_df.merge(captions_df, on='match_key', how='inner')
        
        logger.info(f"✓ Matched {len(merged_df):,} images with captions")
        logger.info(f"  Unmatched images: {len(images_df) - len(merged_df):,}")
        
        # Rename caption column
        merged_df = merged_df.rename(columns={caption_col: 'caption'})
        
        # Select final columns
        final_columns = ['full_path', 'relative_path', 'style', 'filename', 'caption']
        if 'emotion' in merged_df.columns and 'emotion' != caption_col:
            final_columns.append('emotion')
        
        final_columns = [col for col in final_columns if col in merged_df.columns]
        merged_df = merged_df[final_columns]
        
        # Clean captions (limit to max_caption_length words)
        logger.info(f"\n  Cleaning captions (max {self.max_caption_length} words)...")
        merged_df['caption'] = merged_df['caption'].apply(self.clean_caption)
        
        return merged_df
    
    def clean_caption(self, text):
        """
        Clean caption text and limit to max_caption_length words
        Average caption length is <20 words, so 50 is safe
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Limit to max_caption_length words
        words = text.split()
        if len(words) > self.max_caption_length:
            words = words[:self.max_caption_length]
        
        return ' '.join(words)
    
    def create_stratified_subsets(self, df):
        """
        Step 4: Create stratified subsets
        Instructor recommendation: Stratified subsampling by art style
        Instructor recommendation: Progressive dataset scaling (1k, 5k, 10k)
        """
        logger.info(f"\n[STEP 4/6] Creating stratified subsets...")
        logger.info(f"  Preserving art style distribution")
        
        subsets = {}
        
        for size in self.subset_sizes:
            if size > len(df):
                logger.warning(f"  Subset size {size:,} > dataset size {len(df):,}, skipping")
                continue
            
            # Stratified sampling by style
            strata = df['style'].value_counts()
            sampled_dfs = []
            
            for style, count in strata.items():
                # Proportional sampling
                style_size = int(size * count / len(df))
                
                if style_size > 0:
                    style_df = df[df['style'] == style]
                    sample_size = min(style_size, len(style_df))
                    sampled = style_df.sample(n=sample_size, random_state=42)
                    sampled_dfs.append(sampled)
            
            subset_df = pd.concat(sampled_dfs, ignore_index=True)
            subset_df = subset_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            subsets[f'subset_{size}'] = subset_df
            logger.info(f"  ✓ Created subset_{size}: {len(subset_df):,} samples")
        
        return subsets
    
    def build_vocabulary(self, captions):
        """
        Step 5: Build vocabulary with vocab_size limit
        Handles 57,405 raw unique words → 10,000 vocabulary
        """
        logger.info(f"\n[STEP 5/6] Building vocabulary...")
        
        # Tokenize and count words
        word_counts = Counter()
        
        for caption in tqdm(captions, desc="  Tokenizing captions"):
            words = caption.split()  # Already cleaned
            word_counts.update(words)
        
        logger.info(f"  Raw unique words: {len(word_counts):,}")
        
        # Filter by minimum frequency (optional)
        min_freq = 2
        filtered_words = [word for word, count in word_counts.items() 
                         if count >= min_freq]
        logger.info(f"  After min_freq={min_freq}: {len(filtered_words):,}")
        
        # Select top words (reserve 4 slots for special tokens)
        top_words = sorted(filtered_words, 
                          key=lambda x: word_counts[x], 
                          reverse=True)[:self.vocab_size - 4]
        
        # Build vocabulary with special tokens
        vocab = ['<PAD>', '<START>', '<END>', '<UNK>'] + top_words
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for idx, word in enumerate(vocab)}
        
        # Calculate coverage
        covered_words = sum(word_counts[word] for word in top_words)
        total_words = sum(word_counts.values())
        coverage = (covered_words / total_words) * 100
        
        logger.info(f"✓ Final vocabulary size: {len(vocab):,}")
        logger.info(f"  Coverage: {coverage:.2f}% of all words")
        logger.info(f"  Words mapped to <UNK>: {len(word_counts) - len(top_words):,}")
        
        # Show most common words
        logger.info(f"\n  Top 10 most common words:")
        for i, (word, count) in enumerate(word_counts.most_common(10), 1):
            logger.info(f"    {i:2d}. {word:15s}: {count:6,}")
        
        vocab_data = {
            'word2idx': word2idx,
            'idx2word': idx2word,
            'vocab': vocab,
            'vocab_size': len(vocab),
            'max_length': self.max_caption_length,
            'coverage': coverage
        }
        
        return vocab_data
    
    def split_dataset(self, df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Step 6: Split into train/val/test sets
        """
        logger.info(f"\n[STEP 6/6] Splitting into train/val/test sets...")
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]
        
        logger.info(f"  Train: {len(train_df):,} ({train_ratio*100:.0f}%)")
        logger.info(f"  Val:   {len(val_df):,} ({val_ratio*100:.0f}%)")
        logger.info(f"  Test:  {len(test_df):,} ({test_ratio*100:.0f}%)")
        
        return train_df, val_df, test_df
    
    def save_metadata(self, df, filename):
        """Save metadata to CSV"""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"  ✓ Saved {filename}")
    
    def run(self):
        """
        Execute complete preprocessing pipeline
        """
        logger.info("\n" + "="*70)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("="*70)
        
        # Step 1: Scan images
        images_df = self.scan_wikiart_images()
        self.save_metadata(images_df, 'full_image_inventory.csv')
        
        # Step 2: Load captions
        captions_df = self.load_artemis_captions()
        
        # Step 3: Match images to captions
        matched_df = self.match_images_to_captions(images_df, captions_df)
        self.save_metadata(matched_df, 'full_dataset.csv')
        
        # Step 4: Create stratified subsets
        subsets = self.create_stratified_subsets(matched_df)
        for name, subset_df in subsets.items():
            self.save_metadata(subset_df, f'{name}.csv')
            
            # Also create train/val/test splits for each subset
            train_df, val_df, test_df = self.split_dataset(subset_df)
            self.save_metadata(train_df, f'{name}_train.csv')
            self.save_metadata(val_df, f'{name}_val.csv')
            self.save_metadata(test_df, f'{name}_test.csv')
        
        # Step 5: Build vocabulary (from full dataset)
        vocab_data = self.build_vocabulary(matched_df['caption'].tolist())
        
        vocab_path = self.output_dir / 'vocabulary.pkl'
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab_data, f)
        logger.info(f"  ✓ Saved vocabulary.pkl")
        
        # Generate summary report
        self.generate_report(matched_df, subsets, vocab_data)
        
        logger.info("\n" + "="*70)
        logger.info("✓ PREPROCESSING COMPLETE!")
        logger.info("="*70)
        logger.info("\nNext steps:")
        logger.info("  1. Test with small subset:")
        logger.info("     python src/train.py --subset_size 1000 --epochs 5")
        logger.info("  2. Scale up progressively:")
        logger.info("     python src/train.py --subset_size 5000 --epochs 20")
        logger.info("  3. Full training:")
        logger.info("     python src/train.py --subset_size 10000 --epochs 50")
    
    def generate_report(self, full_df, subsets, vocab_data):
        """
        Generate preprocessing summary report
        """
        report_path = self.output_dir / 'preprocessing_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("PREPROCESSING REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write(f"  Image size: {self.image_size}×{self.image_size}\n")
            f.write(f"  Max caption length: {self.max_caption_length} words\n")
            f.write(f"  Vocabulary size: {self.vocab_size}\n\n")
            
            f.write("DATASET STATISTICS:\n")
            f.write(f"  Total matched samples: {len(full_df):,}\n")
            f.write(f"  Art styles: {full_df['style'].nunique()}\n")
            if 'emotion' in full_df.columns:
                f.write(f"  Emotions: {full_df['emotion'].nunique()}\n")
            
            f.write("\n  Caption statistics:\n")
            caption_lengths = full_df['caption'].str.split().str.len()
            f.write(f"    Mean length: {caption_lengths.mean():.2f} words\n")
            f.write(f"    Median length: {caption_lengths.median():.0f} words\n")
            f.write(f"    Max length: {caption_lengths.max()} words\n")
            
            f.write("\n  Style distribution (top 10):\n")
            for style, count in full_df['style'].value_counts().head(10).items():
                pct = (count / len(full_df)) * 100
                f.write(f"    {style:30s}: {count:6,} ({pct:5.2f}%)\n")
            
            f.write("\nVOCABULARY:\n")
            f.write(f"  Raw unique words: 57,405\n")
            f.write(f"  Limited to: {vocab_data['vocab_size']:,}\n")
            f.write(f"  Coverage: {vocab_data['coverage']:.2f}%\n")
            f.write(f"  Special tokens: <PAD>, <START>, <END>, <UNK>\n")
            
            f.write("\nSUBSETS CREATED:\n")
            for name, subset_df in subsets.items():
                f.write(f"  {name}: {len(subset_df):,} samples\n")
                f.write(f"    - {name}_train.csv: 70%\n")
                f.write(f"    - {name}_val.csv: 15%\n")
                f.write(f"    - {name}_test.csv: 15%\n")
            
            f.write("\nFILES CREATED:\n")
            f.write(f"  ✓ full_image_inventory.csv\n")
            f.write(f"  ✓ full_dataset.csv\n")
            for name in subsets.keys():
                f.write(f"  ✓ {name}.csv\n")
                f.write(f"  ✓ {name}_train.csv\n")
                f.write(f"  ✓ {name}_val.csv\n")
                f.write(f"  ✓ {name}_test.csv\n")
            f.write(f"  ✓ vocabulary.pkl\n")
            f.write(f"  ✓ preprocessing_report.txt\n")
        
        logger.info(f"\n  ✓ Generated preprocessing_report.txt")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Unified preprocessing pipeline for WikiArt + ArtEmis'
    )
    parser.add_argument('--wikiart_dir', type=str, default='data/raw/wikiart',
                       help='Path to WikiArt directory')
    parser.add_argument('--captions_file', type=str, 
                       default='data/raw/artemis_captions.csv',
                       help='Path to ArtEmis captions file')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Target image size (default: 224)')
    parser.add_argument('--vocab_size', type=int, default=10000,
                       help='Vocabulary size (default: 10000)')
    parser.add_argument('--max_caption_length', type=int, default=50,
                       help='Maximum caption length in words (default: 50)')
    parser.add_argument('--subset_sizes', type=int, nargs='+', 
                       default=[1000, 5000, 10000],
                       help='Subset sizes to create (default: 1000 5000 10000)')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = UnifiedPreprocessor(
        wikiart_dir=args.wikiart_dir,
        captions_file=args.captions_file,
        output_dir=args.output_dir,
        image_size=args.image_size,
        vocab_size=args.vocab_size,
        max_caption_length=args.max_caption_length,
        subset_sizes=args.subset_sizes
    )
    
    # Run preprocessing
    preprocessor.run()


if __name__ == "__main__":
    main()