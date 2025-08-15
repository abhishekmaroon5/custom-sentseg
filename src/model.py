
import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, x, mask):
        # Get the emission scores from the BiLSTM
        embedded = self.embedding(x)
        outputs, _ = self.lstm(embedded)
        feats = self.hidden2tag(outputs)
        return feats

    def loss(self, feats, mask, tags):
        # Calculate the CRF loss
        return -self.crf(feats, tags, mask=mask, reduction='mean')

    def decode(self, feats, mask):
        # Decode the best path
        return self.crf.decode(feats, mask=mask)

if __name__ == '__main__':
    # Example usage
    VOCAB_SIZE = 1000
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_TAGS = 2 # 0 for non-boundary, 1 for boundary

    model = BiLSTM_CRF(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_TAGS)
    
    # Create a dummy input
    # (batch_size, sequence_length)
    inputs = torch.randint(0, VOCAB_SIZE, (4, 50))
    mask = torch.ones(4, 50, dtype=torch.uint8)
    tags = torch.randint(0, NUM_TAGS, (4, 50))

    # Get emission scores
    feats = model(inputs, mask)
    print("Emission scores shape:", feats.shape)

    # Calculate loss
    loss = model.loss(feats, mask, tags)
    print("Loss:", loss)

    # Decode
    decoded_path = model.decode(feats, mask)
    print("Decoded path:", decoded_path) 