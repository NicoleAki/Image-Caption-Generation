# Image-Caption-Generation
@"
# Data Directory

## ⚠️Large files not included in Git

Data files are too large for GitHub (>100 MB limit).

### To generate the data yourself:

1. Download WikiArt images
2. Download ArtEmis captions  
3. Run: ``python scripts/create_metadata.py``


### Files in Git:
✓ Preprocessing scripts
✓ Small subset files
✓ Vocabulary

### Files NOT in Git:
✗ ``full_dataset.csv`` (too large)
✗ ``full_image_inventory.csv`` (too large)
✗ Raw images (too large)
"@ | Out-File -FilePath data/README.md -Encoding utf8

# Large data files
data/processed/full_dataset.csv
data/processed/full_image_inventory.csv
data/processed/*.csv
data/raw/*
models_pt/*.h5
models_pt/*.pt

## Data Evaluation/Testing witth new data

# 1. Process new test images
python src/preprocessing.py \
    --test_image_dir data/raw/evaluation_images \
    --vocab_file data/processed/vocabulary.pkl \
    --output_dir data/test_processed

# 2. Generate captions using your trained model
python src/inference.py \
    --model_path models/best_cnn_lstm.h5 \
    --test_data data/test_processed/test_images.csv \
    --vocab_file data/processed/vocabulary.pkl \
    --output results/evaluation_captions.csv \
    --beam_width 3

# Step 3: Repeat with other models
python src/inference.py \
    --model_path models/transformer.h5 \
    --test_data data/evaluation_processed/test_images.csv \
    --output results/transformer_captions.csv \
    --top_k 3

## Data Training/ With Wikiart data

# Run preprocessing (replaces both old scripts)
python src/preprocessing.py

# Or with custom settings:
python src/preprocessing.py \
    --wikiart_dir data/raw/wikiart \
    --captions_file data/raw/artemis_captions.csv \
    --image_size 224 \
    --vocab_size 10000 \
    --max_caption_length 50 \
    --subset_sizes 1000 5000 10000