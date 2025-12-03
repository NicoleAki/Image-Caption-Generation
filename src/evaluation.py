import torch
import pandas as pd
import pickle
import argparse
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from models.cnn_lstm import CNN_LSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(path, transform):
    try:
        img = Image.open(path).convert('RGB')
        return transform(img).unsqueeze(0)
    except:
        return None

def main(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load vocabulary
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    df = pd.read_csv(args.test_csv_path)

    # Load model
    model = CNN_LSTM(
        args.embed_size,
        args.hidden_size,
        len(vocab['word2idx']),
        args.num_layers
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    references = []
    hypotheses = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        img_path = row['full_path']
        caption = str(row['caption']).lower().strip()

        image_tensor = load_image(img_path, transform)
        if image_tensor is None:
            continue

        image_tensor = image_tensor.to(device)

        # Generate caption
        with torch.no_grad():
            features = model.encoder(image_tensor)
            sampled_ids = model.decoder.sample(features)

        sampled_ids = sampled_ids[0].cpu().numpy()

        # Convert token IDs → words
        words = []
        for wid in sampled_ids:
            word = vocab['idx2word'][wid]

            # Respect special tokens used in vocabulary
            if word == '<END>':
                break
            if word not in ['<START>', '<PAD>']:
                words.append(word)

        hypotheses.append(words)

        # Reference caption tokens
        references.append([caption.split()])

    # BLEU with smoothing
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
        weights=(1/3, 1/3, 1/3, 0.0),
        smoothing_function=smoothie,
    )
    bleu4 = corpus_bleu(
        references,
        hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie,
    )

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [
        scorer.score(' '.join(ref[0]), ' '.join(hyp))['rougeL'].fmeasure
        for ref, hyp in zip(references, hypotheses)
    ]
    avg_rouge_l = sum(rouge_scores) / len(rouge_scores)

    print("\nEvaluation Results")
    print("------------------")
    print(f"BLEU-1 Score : {bleu1:.4f}")
    print(f"BLEU-2 Score : {bleu2:.4f}")
    print(f"BLEU-3 Score : {bleu3:.4f}")
    print(f"BLEU-4 Score : {bleu4:.4f}")
    print(f"ROUGE-L Score: {avg_rouge_l:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/cnn_lstm.pth')
    parser.add_argument('--vocab_path', type=str, default='data/processed/vocabulary.pkl')
    parser.add_argument('--test_csv_path', type=str, default='data/processed/subset_1000_test.csv')
    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1)
    args = parser.parse_args()

    main(args)
