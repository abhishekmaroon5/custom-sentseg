
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from src.dataset import SentenceSegmentationDataset
from src.model import SentenceSegmentationModel
from tqdm import tqdm
import yaml

def train(config):
    # Load tokenizer, model, and dataset
    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_model_name'])
    model = SentenceSegmentationModel(bert_model_name=config['model']['bert_model_name'])
    dataset = SentenceSegmentationDataset(config['data']['train_file'], tokenizer)
    
    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=int(config['training']['batch_size']), shuffle=True)
    
    # Set up optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(int(config['training']['epochs'])):
        print(f"Epoch {epoch + 1}/{int(config['training']['epochs'])}")
        for batch in tqdm(data_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids, attention_mask)
            
            # Reshape logits and labels for CrossEntropyLoss
            loss = criterion(logits.view(-1, 2), labels.view(-1))
            
            loss.backward()
            optimizer.step()
            
    # Save the model
    torch.save(model.state_dict(), config['model']['save_path'])
    print(f"Model saved to {config['model']['save_path']}")

if __name__ == '__main__':
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    train(config) 