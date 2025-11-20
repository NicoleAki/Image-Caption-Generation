"""
Quick script to check your WikiArt data structure
Run this BEFORE creating metadata to verify everything is set up correctly
"""

import os
from pathlib import Path
from collections import defaultdict

def check_wikiart_structure(wikiart_dir='data/raw/wikiart'):
    """Check WikiArt directory structure"""
    
    print("="*60)
    print("CHECKING WIKIART DATA STRUCTURE")
    print("="*60)
    
    wikiart_path = Path(wikiart_dir)
    
    # Check if directory exists
    if not wikiart_path.exists():
        print(f"\n❌ ERROR: WikiArt directory not found!")
        print(f"   Expected at: {wikiart_path.absolute()}")
        print(f"\n   Please download WikiArt dataset and extract to:")
        print(f"   {wikiart_path.absolute()}")
        return False
    
    print(f"\n✓ Found WikiArt directory: {wikiart_path.absolute()}")
    
    # Check subdirectories (art styles)
    subdirs = [d for d in wikiart_path.iterdir() if d.is_dir()]
    
    if len(subdirs) == 0:
        print(f"\n❌ ERROR: No subdirectories found in WikiArt!")
        print(f"   Expected structure:")
        print(f"   data/raw/wikiart/")
        print(f"   ├── Abstract_Expressionism/")
        print(f"   ├── Impressionism/")
        print(f"   └── ...")
        return False
    
    print(f"\n✓ Found {len(subdirs)} style directories")
    print(f"\n   Style directories found:")
    for i, subdir in enumerate(sorted(subdirs)[:10]):
        print(f"   {i+1}. {subdir.name}")
    if len(subdirs) > 10:
        print(f"   ... and {len(subdirs) - 10} more")
    
    # Count images in each style
    print(f"\n   Checking for images...")
    style_counts = defaultdict(int)
    total_images = 0
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    for style_dir in subdirs:
        count = 0
        for img_path in style_dir.rglob('*'):
            if img_path.suffix in image_extensions:
                count += 1
                total_images += 1
        style_counts[style_dir.name] = count
    
    if total_images == 0:
        print(f"\n❌ ERROR: No images found!")
        print(f"   Check if images were extracted correctly")
        return False
    
    print(f"\n✓ Found {total_images} total images")
    print(f"\n   Images per style (top 10):")
    sorted_styles = sorted(style_counts.items(), key=lambda x: x[1], reverse=True)
    for style, count in sorted_styles[:10]:
        print(f"   {style}: {count} images")
    
    # Show sample file paths
    print(f"\n   Sample image paths:")
    sample_count = 0
    for style_dir in subdirs[:3]:
        for img_path in style_dir.rglob('*'):
            if img_path.suffix in image_extensions:
                print(f"   {img_path.relative_to(wikiart_path)}")
                sample_count += 1
                if sample_count >= 3:
                    break
        if sample_count >= 3:
            break
    
    print(f"\n✓ WikiArt structure looks good!")
    return True


def check_artemis_captions(captions_file='data/raw/artemis_captions.csv'):
    """Check ArtEmis captions file"""
    
    print("\n" + "="*60)
    print("CHECKING ARTEMIS CAPTIONS")
    print("="*60)
    
    captions_path = Path(captions_file)
    
    # Check if file exists
    if not captions_path.exists():
        print(f"\n❌ ERROR: ArtEmis captions file not found!")
        print(f"   Expected at: {captions_path.absolute()}")
        print(f"\n   Possible file names:")
        print(f"   - artemis_captions.csv")
        print(f"   - artemis_dataset.csv")
        print(f"   - artemis.csv")
        print(f"\n   Please download from: https://github.com/optas/artemis")
        
        # Check for alternative files
        possible_names = [
            'data/raw/artemis.csv',
            'data/raw/artemis_dataset.csv',
            'data/raw/artemis_annotations.csv'
        ]
        
        print(f"\n   Checking for alternative files...")
        for alt_file in possible_names:
            if Path(alt_file).exists():
                print(f"   ✓ Found: {alt_file}")
                print(f"   You can use: --captions_file {alt_file}")
                return True
        
        return False
    
    print(f"\n✓ Found captions file: {captions_path.absolute()}")
    print(f"   File size: {captions_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Try to read first few lines
    try:
        import pandas as pd
        df = pd.read_csv(captions_path, nrows=5)
        
        print(f"\n✓ Successfully loaded captions")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"\n   Sample rows:")
        print(df.head())
        
        # Check for required columns
        required_cols = []
        if 'caption' in df.columns or 'utterance' in df.columns:
            print(f"\n✓ Found caption column")
        else:
            print(f"\n⚠️  Warning: No obvious caption column found")
            print(f"   Available columns: {df.columns.tolist()}")
        
        if 'painting' in df.columns or 'image_file' in df.columns:
            print(f"✓ Found image filename column")
        else:
            print(f"\n⚠️  Warning: No obvious image filename column found")
        
    except Exception as e:
        print(f"\n❌ ERROR: Could not read captions file: {e}")
        return False
    
    print(f"\n✓ ArtEmis captions look good!")
    return True


def check_output_directory(output_dir='data/processed'):
    """Check if output directory exists, create if not"""
    
    print("\n" + "="*60)
    print("CHECKING OUTPUT DIRECTORY")
    print("="*60)
    
    output_path = Path(output_dir)
    
    if output_path.exists():
        print(f"\n✓ Output directory exists: {output_path.absolute()}")
        
        # Check for existing files
        existing_files = list(output_path.glob('*.csv')) + list(output_path.glob('*.pkl'))
        if existing_files:
            print(f"\n⚠️  Warning: Found existing files:")
            for f in existing_files:
                print(f"   - {f.name}")
            print(f"\n   These will be overwritten when you create metadata")
    else:
        print(f"\n✓ Output directory will be created: {output_path.absolute()}")
        output_path.mkdir(parents=True, exist_ok=True)
    
    return True


def main():
    """Main check function"""
    
    print("\n" + "="*60)
    print("WIKIART + ARTEMIS DATA STRUCTURE CHECKER")
    print("="*60)
    print("\nThis will verify your data is set up correctly")
    print("before creating metadata.\n")
    
    # Check WikiArt
    wikiart_ok = check_wikiart_structure()
    
    # Check ArtEmis
    artemis_ok = check_artemis_captions()
    
    # Check output directory
    output_ok = check_output_directory()
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if wikiart_ok:
        print("\n✓ WikiArt images: OK")
    else:
        print("\n❌ WikiArt images: NOT FOUND")
    
    if artemis_ok:
        print("✓ ArtEmis captions: OK")
    else:
        print("❌ ArtEmis captions: NOT FOUND")
    
    if output_ok:
        print("✓ Output directory: OK")
    
    if wikiart_ok and artemis_ok:
        print("\n" + "="*60)
        print("✓ READY TO CREATE METADATA!")
        print("="*60)
        print("\nNext step:")
        print("  python scripts/create_metadata.py")
        print("\nOr with custom paths:")
        print("  python scripts/create_metadata.py \\")
        print("    --wikiart_dir data/raw/wikiart \\")
        print("    --captions_file data/raw/artemis_captions.csv")
    else:
        print("\n" + "="*60)
        print("❌ PLEASE FIX ISSUES ABOVE BEFORE PROCEEDING")
        print("="*60)
        
        if not wikiart_ok:
            print("\nTo fix WikiArt:")
            print("  1. Download from: https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view")
            print("  2. Extract to: data/raw/wikiart/")
        
        if not artemis_ok:
            print("\nTo fix ArtEmis:")
            print("  1. Visit: https://github.com/optas/artemis")
            print("  2. Follow their download instructions")
            print("  3. Save captions to: data/raw/artemis_captions.csv")


if __name__ == "__main__":
    main()