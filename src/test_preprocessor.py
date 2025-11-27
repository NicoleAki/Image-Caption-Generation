"""
Process Test/Evaluation Dataset
For use when given new images without captions
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import pickle
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestDatasetProcessor:
    """
    Process test/evaluation dataset (images without captions)
    Uses existing vocabulary from training
    """
    
    def __init__(self,
                 test_image_dir,
                 vocab_file='data/processed/vocabulary.pkl',
                 output_dir='data/test_processed',
                 image_size=224):
        
        self.test_image_dir = Path(test_image_dir)
        self.vocab_file = vocab_file
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*70)
        logger.info("TEST DATASET PROCESSOR")
        logger.info("="*70)
        logger.info(f"Test images: {self.test_image_dir}")
        logger.info(f"Vocabulary: {self.vocab_file}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Image size: {self.image_size}×{self.image_size}")
    
    def load_vocabulary(self):
        """Load pre-trained vocabulary"""
        logger.info("\nLoading vocabulary from training...")
        
        with open(self.vocab_file, 'rb') as f:
            vocab_data = pickle.load(f)
        
        logger.info(f"✓ Loaded vocabulary with {vocab_data['vocab_size']:,} words")
        return vocab_data
    
    def validate_image(self, img_path):
        """Validate image file"""
        try:
            with Image.open(img_path) as img:
                img.verify()
            return True
        except Exception as e:
            logger.warning(f"Corrupt image: {img_path}")
            return False
    
    def scan_test_images(self):
        """
        Scan test image directory
        Supports flat structure or nested folders
        """
        logger.info("\nScanning test images...")
        
        image_records = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
        # Check if directory has subdirectories or flat structure
        subdirs = [d for d in self.test_image_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            # Nested structure (e.g., by style/artist)
            logger.info("  Detected nested folder structure")
            for subdir in tqdm(subdirs, desc="Scanning folders"):
                category = subdir.name
                
                for img_path in subdir.rglob('*'):
                    if img_path.suffix in image_extensions:
                        if self.validate_image(img_path):
                            record = {
                                'full_path': str(img_path),
                                'relative_path': str(img_path.relative_to(self.test_image_dir)),
                                'filename': img_path.name,
                                'category': category
                            }
                            image_records.append(record)
        else:
            # Flat structure
            logger.info("  Detected flat folder structure")
            for img_path in tqdm(list(self.test_image_dir.glob('*')), 
                                desc="Scanning images"):
                if img_path.suffix in image_extensions:
                    if self.validate_image(img_path):
                        record = {
                            'full_path': str(img_path),
                            'relative_path': img_path.name,
                            'filename': img_path.name,
                            'category': 'unknown'
                        }
                        image_records.append(record)
        
        test_df = pd.DataFrame(image_records)
        logger.info(f"✓ Found {len(test_df):,} valid test images")
        
        if 'category' in test_df.columns:
            n_categories = test_df['category'].nunique()
            if n_categories > 1:
                logger.info(f"  Categories: {n_categories}")
                for cat, count in test_df['category'].value_counts().head(5).items():
                    logger.info(f"    {cat}: {count:,}")
        
        return test_df
    
    def process(self):
        """
        Process test dataset
        """
        logger.info("\n" + "="*70)
        logger.info("PROCESSING TEST DATASET")
        logger.info("="*70)
        
        # Load vocabulary (for compatibility)
        vocab_data = self.load_vocabulary()
        
        # Scan test images
        test_df = self.scan_test_images()
        
        # Save metadata
        output_file = self.output_dir / 'test_images.csv'
        test_df.to_csv(output_file, index=False)
        logger.info(f"\n✓ Saved test metadata: {output_file}")
        
        # Save vocab info for reference
        vocab_info_file = self.output_dir / 'vocab_info.txt'
        with open(vocab_info_file, 'w') as f:
            f.write(f"Vocabulary size: {vocab_data['vocab_size']}\n")
            f.write(f"Max caption length: {vocab_data.get('max_length', 50)}\n")
            f.write(f"Image size: {self.image_size}×{self.image_size}\n")
        
        logger.info(f"✓ Saved vocabulary info: {vocab_info_file}")
        
        # Generate summary
        self.generate_summary(test_df, vocab_data)
        
        logger.info("\n" + "="*70)
        logger.info("✓ TEST DATASET PROCESSING COMPLETE!")
        logger.info("="*70)
        logger.info("\nNext step: Generate captions")
        logger.info("  python src/generate_captions.py \\")
        logger.info("    --model_path models/best_model.h5 \\")
        logger.info("    --test_images data/test_processed/test_images.csv \\")
        logger.info("    --output results/generated_captions.csv")
    
    def generate_summary(self, test_df, vocab_data):
        """Generate processing summary"""
        summary_file = self.output_dir / 'test_processing_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TEST DATASET PROCESSING SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write(f"  Test image directory: {self.test_image_dir}\n")
            f.write(f"  Image size: {self.image_size}×{self.image_size}\n")
            f.write(f"  Vocabulary file: {self.vocab_file}\n")
            f.write(f"  Vocabulary size: {vocab_data['vocab_size']:,}\n\n")
            
            f.write("TEST DATASET:\n")
            f.write(f"  Total images: {len(test_df):,}\n")
            
            if 'category' in test_df.columns:
                n_cat = test_df['category'].nunique()
                if n_cat > 1:
                    f.write(f"  Categories: {n_cat}\n\n")
                    f.write("  Distribution:\n")
                    for cat, count in test_df['category'].value_counts().items():
                        pct = (count / len(test_df)) * 100
                        f.write(f"    {cat:30s}: {count:5,} ({pct:5.2f}%)\n")
            
            f.write("\nOUTPUT FILES:\n")
            f.write("  ✓ test_images.csv - Test image metadata\n")
            f.write("  ✓ vocab_info.txt - Vocabulary information\n")
            f.write("  ✓ test_processing_summary.txt - This file\n")
            
            f.write("\nUSAGE:\n")
            f.write("  Use test_images.csv for caption generation\n")
            f.write("  Images will be loaded on-the-fly during inference\n")
        
        logger.info(f"✓ Generated summary: {summary_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process test/evaluation dataset'
    )
    parser.add_argument('--test_image_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--vocab_file', type=str, 
                       default='data/processed/vocabulary.pkl',
                       help='Path to vocabulary file from training')
    parser.add_argument('--output_dir', type=str, 
                       default='data/test_processed',
                       help='Output directory for processed test data')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Target image size (should match training)')
    
    args = parser.parse_args()
    
    # Create processor
    processor = TestDatasetProcessor(
        test_image_dir=args.test_image_dir,
        vocab_file=args.vocab_file,
        output_dir=args.output_dir,
        image_size=args.image_size
    )
    
    # Process test data
    processor.process()


if __name__ == "__main__":
    main()