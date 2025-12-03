import argparse
import pickle

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm

from models.transformer import VisionLanguageTransformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path, transform):
    try:
        img = Image.open(path).convert("RGB")
        return transform(img).unsqueeze(0)
    except Exception:
        return None


def main(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load vocab
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)
    word2idx = vocab["word2idx"]
    idx2word = vocab["idx2word"]
    vocab_size = len(word2idx)

    # Load data
    df = pd.read_csv(args.test_csv_path)

    # Build model (hyperparams must match train_transformer.py)
    model = VisionLanguageTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        max_seq_length=args.max_seq_length,
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    references = []
    hypotheses = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Transformer"):
        img_path = row["full_path"]
        caption = str(row["caption"]).lower().strip()

        image_tensor = load_image(img_path, transform)
        if image_tensor is None:
            continue
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            sampled_ids = model.sample(image_tensor)

        sampled_ids = sampled_ids[0].cpu().numpy()

        words = []
        for wid in sampled_ids:
            word = idx2word[wid]
            if word == "<END>":
                break
            if word in ("<START>", "<PAD>", "<UNK>"):
                continue
            words.append(word)

        hypotheses.append(words)
        references.append([caption.split()])

    smoothie = SmoothingFunction().method1

    bleu1 = corpus_bleu(
        references,
        hypotheses,
        weights=(1.0, 0.0, 0.0, 0.0),
        smoothing_function=smoothie,
    )
    bleu2 = corpus_bleu(
        references,
        hypotheses,
        weights=(0.5, 0.5, 0.0, 0.0),
        smoothing_function=smoothie,
    )
    bleu3 = corpus_bleu(
        references,
        hypotheses,
        weights=(1 / 3, 1 / 3, 1 / 3, 0.0),
        smoothing_function=smoothie,
    )
    bleu4 = corpus_bleu(
        references,
        hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie,
    )

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [
        scorer.score(" ".join(ref[0]), " ".join(hyp))["rougeL"].fmeasure
        for ref, hyp in zip(references, hypotheses)
    ]
    avg_rouge_l = sum(rouge_scores) / len(rouge_scores)

    print("\nTransformer Evaluation Results")
    print("-----------------------------")
    print(f"BLEU-1 Score : {bleu1:.4f}")
    print(f"BLEU-2 Score : {bleu2:.4f}")
    print(f"BLEU-3 Score : {bleu3:.4f}")
    print(f"BLEU-4 Score : {bleu4:.4f}")
    print(f"ROUGE-L Score: {avg_rouge_l:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/transformer.pth")
    parser.add_argument("--vocab_path", type=str, default="data/processed/vocabulary.pkl")
    parser.add_argument("--test_csv_path", type=str, default="data/processed/full_dataset_test.csv")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--max_seq_length", type=int, default=50)
    args = parser.parse_args()

    main(args)
