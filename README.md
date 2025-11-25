# Image-Caption-Generation
@"
# Data Directory

## ⚠️Large files not included in Git

Data files are too large for GitHub (>100 MB limit).

### To generate the data yourself:

1. Download WikiArt images
2. Download ArtEmis captions  
3. Run: ``python scripts/create_metadata.py``

This creates:
- ``full_dataset.csv`` (130 MB) - local only
- ``subset_*.csv`` (small) - committed to Git
- ``vocabulary.pkl`` - committed to Git

### Files in Git:
✓ Preprocessing scripts
✓ Small subset files
✓ Vocabulary

### Files NOT in Git:
✗ ``full_dataset.csv`` (too large)
✗ ``full_image_inventory.csv`` (too large)
✗ Raw images (too large)
"@ | Out-File -FilePath data/README.md -Encoding utf8

git add data/README.md
git commit -m "Add data README with instructions"
git push origin main
```

### **Prevent Future Issues:**

Your `.gitignore` should now have:
```
# Large data files
data/processed/full_dataset.csv
data/processed/full_image_inventory.csv
data/processed/*.csv
data/raw/*
models_pt/*.h5
models_pt/*.pt