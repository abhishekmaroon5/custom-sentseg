
import torch
from torch.utils.data import DataLoader
from src.dataset import SentenceSegmentationDataset_CRF
from src.model import BiLSTM_CRF
from tqdm import tqdm
import yaml

def train(config):
    # Load dataset and create vocabulary
    dataset = SentenceSegmentationDataset_CRF(config['data']['train_file'])
    
    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=dataset.collate_fn)
    
    # Initialize model
    model = BiLSTM_CRF(
        vocab_size=len(dataset.word_to_idx),
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_tags=2
    )
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    
    # Training loop
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(int(config['training']['epochs'])):
        print(f"Epoch {epoch + 1}/{int(config['training']['epochs'])}")
        for sentences, tags, masks in tqdm(data_loader, desc="Training"):
            sentences, tags, masks = sentences.to(device), tags.to(device), masks.to(device).byte()
            
            optimizer.zero_grad()
            
            feats = model(sentences, masks)
            loss = model.loss(feats, masks, tags)
            
            loss.backward()
            optimizer.step()
            
    # Save the model
    torch.save(model.state_dict(), config['model']['save_path'])
    print(f"Model saved to {config['model']['save_path']}")
    
    # Save the vocabulary
    torch.save(dataset.word_to_idx, 'models/word_to_idx.pt')
    print(f"Vocabulary saved to models/word_to_idx.pt")

if __name__ == '__main__':
    # Load configuration
    with open('configs/config_crf.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    train(config) 