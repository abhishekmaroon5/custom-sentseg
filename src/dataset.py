
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class SentenceSegmentationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.labels = []

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Split the text into segments that fit within the max_length
        # This is a simple approach. A more sophisticated approach
        # would be to handle sentence boundaries more carefully.
        words = text.split()
        for i in range(0, len(words), max_length - 2): # -2 for [CLS] and [SEP]
            segment_words = words[i : i + max_length - 2]
            
            # Create labels: 1 if the word is <eos>, 0 otherwise
            segment_labels = [1 if word == "<eos>" else 0 for word in segment_words]
            
            # Reconstruct the text for tokenization
            segment_text = " ".join(segment_words)
            
            # Tokenize the text
            tokenized_input = self.tokenizer(
                segment_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = tokenized_input["input_ids"].squeeze()
            attention_mask = tokenized_input["attention_mask"].squeeze()
            
            # Align labels with tokens
            aligned_labels = self.align_labels_with_tokens(segment_words, segment_labels, input_ids)

            self.inputs.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })
            self.labels.append(aligned_labels)

    def align_labels_with_tokens(self, words, labels, token_ids):
        aligned_labels = torch.zeros(self.max_length, dtype=torch.long)
        word_idx = 0
        for i, token_id in enumerate(token_ids):
            if token_id == self.tokenizer.cls_token_id or token_id == self.tokenizer.sep_token_id or token_id == self.tokenizer.pad_token_id:
                continue

            token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            
            # This is a simplified alignment. It might not be perfect.
            # A more robust solution would use the word_ids from the tokenizer.
            if token.startswith("##"):
                # This is a subword token, it gets the same label as the previous token
                # which corresponds to the start of the word.
                # However, for simplicity, we are not handling this perfectly here.
                # We'll just assign 0.
                pass 
            else:
                if word_idx < len(labels):
                    aligned_labels[i] = labels[word_idx]
                    word_idx += 1
        return aligned_labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx]["input_ids"],
            "attention_mask": self.inputs[idx]["attention_mask"],
            "labels": self.labels[idx]
        }

if __name__ == '__main__':
    # Example usage
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = SentenceSegmentationDataset('data/train.txt', tokenizer)
    
    # Print a sample
    sample = dataset[0]
    print("Input IDs:", sample["input_ids"])
    print("Attention Mask:", sample["attention_mask"])
    print("Labels:", sample["labels"])
    print("Decoded Tokens:", tokenizer.convert_ids_to_tokens(sample["input_ids"]))

class SentenceSegmentationDataset_CRF(Dataset):
    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Build vocabulary
        self.words = sorted(list(set(text.split())))
        self.word_to_idx = {word: i+2 for i, word in enumerate(self.words)}
        self.word_to_idx["<pad>"] = 0
        self.word_to_idx["<unk>"] = 1

        # Prepare sequences and labels
        self.sequences = []
        self.labels = []
        
        sentences = text.split('<eos>')
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            words = sentence.strip().split()
            if not words:
                continue

            # Create sequence of word indices
            seq = [self.word_to_idx.get(word, self.word_to_idx["<unk>"]) for word in words]
            
            # Create labels: 1 for the last word of a sentence, 0 otherwise
            lbls = [0] * (len(words) -1) + [1]

            self.sequences.append(torch.tensor(seq, dtype=torch.long))
            self.labels.append(torch.tensor(lbls, dtype=torch.long))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
        
    def collate_fn(self, batch):
        sequences, labels = zip(*batch)
        
        # Pad sequences and labels
        padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.word_to_idx["<pad>"])
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0) # pad labels with 0
        
        # Create masks
        masks = (padded_sequences != self.word_to_idx["<pad>"])
        
        return padded_sequences, padded_labels, masks 