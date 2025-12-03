import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import argparse
from models.cnn_lstm import CNN_LSTM

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')

    if transform is not None:
        image = transform(image).unsqueeze(0)  # add batch dim

    return image

def main(args):
    # Image preprocessing (must match training!)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Load vocabulary
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build model
    model = CNN_LSTM(
        args.embed_size,
        args.hidden_size,
        len(vocab['word2idx']),
        args.num_layers
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Prepare image
    image_tensor = load_image(args.image_path, transform).to(device)
    
    # Encode → Decode
    features = model.encoder(image_tensor)
    sampled_ids = model.decoder.sample(features)

    # Convert word IDs → sentence
    sampled_ids = sampled_ids[0].cpu().numpy()

    words = []
    for word_id in sampled_ids:
        word = vocab['idx2word'][word_id]
        # Respect special tokens used in preprocessing/vocabulary
        if word == '<END>':
            break
        if word in ('<START>', '<PAD>', '<UNK>'):
            continue
        words.append(word)

    caption = ' '.join(words)
    print("\nGenerated Caption:")
    print(caption)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='models/cnn_lstm.pth')
    parser.add_argument('--vocab_path', type=str, default='data/processed/vocabulary.pkl')
    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1)
    args = parser.parse_args()
    main(args)
