
import torch
from src.model import BiLSTM_CRF
import argparse

def segment_text(text, model, word_to_idx):
    # Preprocess the text
    text = text.replace('.', ' <eos> ')
    words = text.strip().split()
    sequence = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in words]
    sequence = torch.tensor([sequence], dtype=torch.long)
    mask = torch.ones_like(sequence, dtype=torch.uint8)
    
    # Run the model
    model.eval()
    with torch.no_grad():
        feats = model(sequence, mask)
        decoded_path = model.decode(feats, mask)
    
    # Reconstruct sentences
    sentences = []
    current_sentence = []
    for word, tag in zip(words, decoded_path[0]):
        if word == "<eos>":
            if current_sentence:
                sentences.append(" ".join(current_sentence) + ".")
                current_sentence = []
        else:
            current_sentence.append(word)
    
    if current_sentence:
        sentences.append(" ".join(current_sentence))
        
    return sentences

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sentence Segmentation Inference")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--vocab_path', type=str, required=True, help="Path to the vocabulary")
    parser.add_argument('--text', type=str, required=True, help="Text to segment")
    args = parser.parse_args()

    # Load the model and vocabulary
    word_to_idx = torch.load(args.vocab_path)
    model = BiLSTM_CRF(
        vocab_size=len(word_to_idx),
        embedding_dim=128, # Should match the training config
        hidden_dim=256,    # Should match the training config
        num_tags=2
    )
    model.load_state_dict(torch.load(args.model_path))
    
    # Segment the text
    segmented_sentences = segment_text(args.text, model, word_to_idx)
    
    # Print the result
    for sentence in segmented_sentences:
        print(sentence) 