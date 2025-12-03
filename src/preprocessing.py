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

class EnhancedPreprocessor:
    """
    Enhanced preprocessing with stop words preservation and rare word handling
    """
    
    def __init__(self, 
                 wikiart_dir='data/raw/wikiart',
                 captions_file='data/raw/artemis_captions.csv',
                 output_dir='data/processed',
                 image_size=224,
                 vocab_size=10000,
                 max_caption_length=50,
                 subset_sizes=[1000, 5000, 10000],
                 min_word_frequency=5):
        
        self.wikiart_dir = Path(wikiart_dir)
        self.captions_file = captions_file
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.vocab_size = vocab_size
        self.max_caption_length = max_caption_length
        self.subset_sizes = subset_sizes
        self.min_word_frequency = min_word_frequency
        
        # Comprehensive stop words list
        self.STOP_WORDS = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 
            'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 
            'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 
            'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'some', 'such', 
            'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just'
        }
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_output_dir = self.output_dir / 'resized_images'
        self.images_output_dir.mkdir(exist_ok=True)

        logger.info("="*70)
        logger.info("ENHANCED PREPROCESSING PIPELINE")
        logger.info("="*70)
        logger.info(f"Configuration:")
        logger.info(f"  WikiArt directory: {self.wikiart_dir}")
        logger.info(f"  Captions file: {self.captions_file}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Image size: {self.image_size}×{self.image_size}")
        logger.info(f"  Vocabulary size: {self.vocab_size}")
        logger.info(f"  Max caption length: {self.max_caption_length} words")
        logger.info(f"  Min word frequency: {self.min_word_frequency}")
        logger.info(f"  Subset sizes: {self.subset_sizes}")
        logger.info(f"  Stop words: PRESERVED (essential for grammar)")
    
    def validate_and_resize_image(self, img_path):
        try:
            with Image.open(img_path) as img:
                img.verify()
            return self.resize_image(img_path)
        except Exception as e:
            logger.warning(f"Corrupt image: {img_path} - {e}")
            return False

    def resize_image(self, img_path):
        try:
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img_resized = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
                
                # Save resized image
                relative_path = Path(img_path).relative_to(self.wikiart_dir)
                output_path = self.images_output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                img_resized.save(output_path, 'JPEG', quality=90)
                return True
        except Exception as e:
            logger.warning(f"Failed to resize image: {img_path} : {e}")
            return False
    
    def scan_wikiart_images(self):

        logger.info("\n[STEP 1/6] Scanning WikiArt images and resizing...")
        
        image_records = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
        for style_folder in tqdm(list(self.wikiart_dir.iterdir()), 
                                desc="Scanning and resizing styles"):
            if not style_folder.is_dir():
                continue
            
            style_name = style_folder.name
            
            for img_path in style_folder.rglob('*'):
                if img_path.suffix in image_extensions:
                    if self.validate_and_resize_image(img_path):
                        relative_path = img_path.relative_to(self.wikiart_dir)
                        resized_path = self.images_output_dir / relative_path
                        
                        record = {
                            'full_path': str(resized_path),
                            'original_path': str(img_path),
                            'relative_path': str(relative_path),
                            'style': style_name,
                            'filename': img_path.name
                        }
                        image_records.append(record)
        
        images_df = pd.DataFrame(image_records)
        logger.info(f"✓ Found and resized {len(images_df):,} valid images across "
                   f"{images_df['style'].nunique()} styles")
        
        logger.info("\n  Top 10 styles by count:")
        for style, count in images_df['style'].value_counts().head(10).items():
            logger.info(f"    {style}: {count:,}")
        
        return images_df
    
    def load_artemis_captions(self):
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
        
        # Create matching keys
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
        final_columns = ['full_path', 'original_path', 'relative_path', 'style', 'filename', 'caption']
        if 'emotion' in merged_df.columns and 'emotion' != caption_col:
            final_columns.append('emotion')
        
        final_columns = [col for col in final_columns if col in merged_df.columns]
        merged_df = merged_df[final_columns]
        
        # Clean captions
        logger.info(f"\n  Cleaning captions (max {self.max_caption_length} words)...")
        merged_df['caption'] = merged_df['caption'].apply(self.clean_caption)
        
        return merged_df
    
    def clean_caption(self, text):
        """
        Clean caption text - PRESERVES STOP WORDS
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase but preserve grammatical structure
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
    
    def build_vocabulary(self, captions):
        """
        Enhanced vocabulary building with stop words preservation and rare word handling
        """
        logger.info(f"\n[STEP 4/6] Building enhanced vocabulary...")
        logger.info(f"  Strategy: Keep stop words, map words with frequency ≤ {self.min_word_frequency} to <UNK>")
        
        # Tokenize and count words
        word_counts = Counter()
        
        for caption in tqdm(captions, desc="  Tokenizing captions"):
            words = caption.split()  # Already cleaned
            word_counts.update(words)
        
        logger.info(f"  Raw unique words: {len(word_counts):,}")
        
        # Separate stop words and content words
        stop_words_in_corpus = {word: count for word, count in word_counts.items() 
                               if word in self.STOP_WORDS}
        content_words = {word: count for word, count in word_counts.items() 
                        if word not in self.STOP_WORDS}
        
        logger.info(f"  Stop words in corpus: {len(stop_words_in_corpus):,}")
        logger.info(f"  Content words: {len(content_words):,}")
        
        # Filter content words by minimum frequency
        frequent_content_words = {word: count for word, count in content_words.items() 
                                 if count > self.min_word_frequency}
        rare_content_words = {word: count for word, count in content_words.items() 
                             if count <= self.min_word_frequency}
        
        logger.info(f"  Content words with frequency > {self.min_word_frequency}: {len(frequent_content_words):,}")
        logger.info(f"  Content words with frequency ≤ {self.min_word_frequency}: {len(rare_content_words):,}")
        
        # Combine stop words and frequent content words
        candidate_words = list(stop_words_in_corpus.keys()) + list(frequent_content_words.keys())
        
        # Sort by frequency and select top words
        candidate_words_sorted = sorted(candidate_words, 
                                      key=lambda x: word_counts[x], 
                                      reverse=True)
        
        # Take top words (reserve 4 slots for special tokens)
        available_slots = self.vocab_size - 4
        if len(candidate_words_sorted) > available_slots:
            selected_words = candidate_words_sorted[:available_slots]
            logger.info(f"  Limited vocabulary from {len(candidate_words_sorted):,} to {available_slots:,} words")
        else:
            selected_words = candidate_words_sorted
            logger.info(f"  Using all {len(selected_words):,} candidate words")
        
        # Build vocabulary with special tokens
        vocab = ['<PAD>', '<START>', '<END>', '<UNK>'] + selected_words
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for idx, word in enumerate(vocab)}
        
        # Calculate coverage
        covered_words = sum(word_counts[word] for word in selected_words)
        total_words = sum(word_counts.values())
        coverage = (covered_words / total_words) * 100
        
        # Calculate what gets mapped to <UNK>
        unk_words_count = sum(count for word, count in rare_content_words.items())
        unk_coverage = (unk_words_count / total_words) * 100
        
        logger.info(f"✓ Final vocabulary size: {len(vocab):,}")
        logger.info(f"  Coverage: {coverage:.2f}% of all words")
        logger.info(f"  Words mapped to <UNK>: {len(rare_content_words):,} ({unk_coverage:.2f}% of corpus)")
        logger.info(f"  Stop words preserved: {len(stop_words_in_corpus):,}")
        
        # Show most common words
        logger.info(f"\n  Top 15 most common words:")
        for i, (word, count) in enumerate(word_counts.most_common(15), 1):
            word_type = "STOP" if word in self.STOP_WORDS else "CONTENT"
            logger.info(f"    {i:2d}. {word:15s}: {count:6,} ({word_type})")
        
        vocab_data = {
            'word2idx': word2idx,
            'idx2word': idx2word,
            'vocab': vocab,
            'vocab_size': len(vocab),
            'max_length': self.max_caption_length,
            'coverage': coverage,
            'unk_coverage': unk_coverage,
            'stop_words_count': len(stop_words_in_corpus),
            'min_frequency': self.min_word_frequency
        }
        
        return vocab_data

    def create_stratified_subsets(self, df):
        logger.info(f"\n[STEP 5/6] Creating stratified subsets...")
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

    def split_dataset(self, df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):

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
        logger.info("STARTING ENHANCED PREPROCESSING PIPELINE")
        logger.info("="*70)
        
        # Step 1: Scan and resize images
        images_df = self.scan_wikiart_images()
        self.save_metadata(images_df, 'full_image_inventory.csv')
        
        # Step 2: Load captions
        captions_df = self.load_artemis_captions()
        
        # Step 3: Match images to captions
        matched_df = self.match_images_to_captions(images_df, captions_df)
        self.save_metadata(matched_df, 'full_dataset.csv')
        
        # Step 4: Build enhanced vocabulary
        vocab_data = self.build_vocabulary(matched_df['caption'].tolist())
        
        vocab_path = self.output_dir / 'vocabulary.pkl'
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab_data, f)
        logger.info(f"  ✓ Saved enhanced vocabulary.pkl")
        
        # Step 5: Create stratified subsets
        subsets = self.create_stratified_subsets(matched_df)
        for name, subset_df in subsets.items():
            self.save_metadata(subset_df, f'{name}.csv')
            
            # Create train/val/test splits for each subset
            train_df, val_df, test_df = self.split_dataset(subset_df)
            self.save_metadata(train_df, f'{name}_train.csv')
            self.save_metadata(val_df, f'{name}_val.csv')
            self.save_metadata(test_df, f'{name}_test.csv')
        
        # Generate summary report
        self.generate_report(matched_df, subsets, vocab_data)
        
        logger.info("\n" + "="*70)
        logger.info("✓ ENHANCED PREPROCESSING COMPLETE!")
        logger.info("="*70)

    def generate_report(self, full_df, subsets, vocab_data):
        """
        Generate enhanced preprocessing summary report
        """
        report_path = self.output_dir / 'enhanced_preprocessing_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ENHANCED PREPROCESSING REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("VOCABULARY STRATEGY:\n")
            f.write(f"  Total vocabulary size: {vocab_data['vocab_size']:,}\n")
            f.write(f"  Stop words preserved: {vocab_data['stop_words_count']:,}\n")
            f.write(f"  Minimum frequency threshold: {vocab_data['min_frequency']}\n")
            f.write(f"  Text coverage: {vocab_data['coverage']:.2f}%\n")
            f.write(f"  <UNK> coverage: {vocab_data['unk_coverage']:.2f}%\n\n")
            
            f.write("KEY FEATURES:\n")
            f.write("  1. Stop words preserved for grammatical correctness\n")
            f.write("  2. Rare words (frequency ≤5) mapped to <UNK>\n")
            f.write("  3. Personal/emotional bigrams preserved ('looks like', 'makes me')\n")
            f.write("  4. Natural language patterns maintained\n\n")
            
            f.write("DATASET STATISTICS:\n")
            f.write(f"  Total matched samples: {len(full_df):,}\n")
            f.write(f"  Art styles: {full_df['style'].nunique()}\n")
            
            if 'emotion' in full_df.columns:
                f.write(f"  Emotions: {full_df['emotion'].nunique()}\n")
            
            f.write("\nSUBSETS CREATED:\n")
            for name, subset_df in subsets.items():
                f.write(f"  {name}: {len(subset_df):,} samples\n")
        
        logger.info(f"\n  ✓ Generated enhanced_preprocessing_report.txt")

# Test Preprocessing for Unseen Dataset
class InferencePreprocessor:
    """
    Preprocessing for unseen datasets without captions
    """
    
    def __init__(self, vocabulary_path, image_size=224):
        self.image_size = image_size
        self.vocabulary_path = vocabulary_path
        
        # Load vocabulary
        with open(vocabulary_path, 'rb') as f:
            self.vocab_data = pickle.load(f)
        
        self.word2idx = self.vocab_data['word2idx']
        logger.info(f"Loaded vocabulary with {len(self.word2idx):,} words")
    
    def preprocess_single_image(self, image_path, output_dir):
        """
        Preprocess a single image for inference
        """
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                img_resized = img.resize((self.image_size, self.image_size), 
                                       Image.Resampling.LANCZOS)
                
                # Save to output directory
                output_path = Path(output_dir) / Path(image_path).name
                img_resized.save(output_path, 'JPEG', quality=90)
                
                return {
                    'success': True,
                    'output_path': str(output_path),
                    'original_size': img.size,
                    'resized_size': (self.image_size, self.image_size)
                }
                
        except Exception as e:
            logger.error(f"Failed to preprocess {image_path}: {e}")
            return {'success': False, 'error': str(e)}
    
    def preprocess_image_directory(self, input_dir, output_dir):
        """
        Preprocess all images in a directory for inference
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        results = []
        
        logger.info(f"Preprocessing images from {input_dir}")
        
        for img_path in tqdm(list(input_dir.rglob('*')), desc="Preprocessing images"):
            if img_path.suffix.lower() in image_extensions:
                result = self.preprocess_single_image(img_path, output_dir)
                result['original_path'] = str(img_path)
                results.append(result)
        
        successful = sum(1 for r in results if r['success'])
        logger.info(f"✓ Preprocessed {successful}/{len(results)} images successfully")
        
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced preprocessing pipeline for WikiArt + ArtEmis'
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
    parser.add_argument('--min_word_frequency', type=int, default=5,
                       help='Minimum word frequency (words ≤ this become <UNK>)')
    parser.add_argument('--subset_sizes', type=int, nargs='+', 
                       default=[1000, 5000, 10000],
                       help='Subset sizes to create (default: 1000 5000 10000)')
    
    args = parser.parse_args()
    
    # Create enhanced preprocessor
    preprocessor = EnhancedPreprocessor(
        wikiart_dir=args.wikiart_dir,
        captions_file=args.captions_file,
        output_dir=args.output_dir,
        image_size=args.image_size,
        min_word_frequency=args.min_word_frequency
    )
    
    # Run preprocessing
    preprocessor.run()

    # Also split the cleaned 10k dataset if it exists
    cleaned_10k_path = Path(args.output_dir) / 'subset_10000_clean.csv'
    if cleaned_10k_path.exists():
        print(f"\nFound {cleaned_10k_path}, splitting into train/val/test...")
        df_10k_clean = pd.read_csv(cleaned_10k_path)
        train_df, val_df, test_df = preprocessor.split_dataset(df_10k_clean)
        preprocessor.save_metadata(train_df, 'subset_10000_clean_train.csv')
        preprocessor.save_metadata(val_df, 'subset_10000_clean_val.csv')
        preprocessor.save_metadata(test_df, 'subset_10000_clean_test.csv')

if __name__ == "__main__":
    main()