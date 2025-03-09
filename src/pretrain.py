import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm
import os
import json
from vqa_model import MedicalVQAModel
from dataset import MedicalVQADataset

def load_medical_vocabulary(file_path="e:/MedVQA/data/medical_terms.json"):
    """Load medical terms and convert to token IDs"""
    if not os.path.exists(file_path):
        print(f"Medical terms file not found at {file_path}. Creating a sample file...")
        # Create a sample file with some medical terms
        medical_terms = [
            "diagnosis", "prognosis", "lesion", "tumor", "carcinoma", "biopsy",
            "pathology", "radiology", "metastasis", "benign", "malignant",
            "lymph", "node", "inflammation", "edema", "necrosis", "atrophy"
        ]
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(medical_terms, f)
    
    with open(file_path, 'r') as f:
        medical_terms = json.load(f)
    
    # Convert terms to token IDs using BioGPT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
    term_ids = []
    
    for term in medical_terms:
        ids = tokenizer.encode(term, add_special_tokens=False)
        term_ids.extend(ids)
    
    return torch.tensor(list(set(term_ids)))  # Remove duplicates

def pretrain_contrastive(
    batch_size=32,
    num_epochs=10,
    learning_rate=2e-5,
    warmup_steps=1000,
    output_dir="e:/MedVQA/models",
    device=None,
    num_experts=4
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create model with MoE and KV Cache
    model = MedicalVQAModel(num_experts=num_experts)
    
    # Load medical vocabulary
    medical_term_ids = load_medical_vocabulary()
    model.set_medical_term_ids(medical_term_ids.to(device))
    
    model.to(device)
    
    # Create datasets
    train_dataset = MedicalVQADataset(split='train')
    val_dataset = MedicalVQADataset(split='validation')
    
    # Create data loaders with a custom collate function to handle PIL images
    def collate_fn(batch):
        # Filter out any None values
        batch = [item for item in batch if item is not None]
        if not batch:
            return {}
            
        # Create a batch dictionary
        batch_dict = {
            'image': torch.stack([item['image'] for item in batch]),
            'question_ids': torch.stack([item['question_ids'] for item in batch]),
            'question_mask': torch.stack([item['question_mask'] for item in batch]),
            'is_yes_no': torch.tensor([item['is_yes_no'] for item in batch]),
        }
        
        # Handle answers based on question type
        answers = []
        for item in batch:
            answers.append(item['answer'])
        
        # Convert answers to tensor
        if all(isinstance(ans, int) for ans in answers):
            batch_dict['answer'] = torch.tensor(answers)
        elif all(isinstance(ans, torch.Tensor) for ans in answers):
            batch_dict['answer'] = torch.stack(answers)
        else:
            # Mixed types - handle separately
            batch_dict['answer'] = answers
            
        return batch_dict
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Skip empty batches
            if not batch:
                continue
                
            # Move batch to device
            images = batch['image'].to(device)
            question_ids = batch['question_ids'].to(device)
            question_mask = batch['question_mask'].to(device)
            
            # Forward pass in contrastive mode
            outputs = model(images, {'input_ids': question_ids, 'attention_mask': question_mask}, mode='contrastive')
            
            # Compute contrastive loss
            loss = model.contrastive_loss(
                outputs['image_embeddings'],
                outputs['text_embeddings']
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Skip empty batches
                if not batch:
                    continue
                    
                # Move batch to device
                images = batch['image'].to(device)
                question_ids = batch['question_ids'].to(device)
                question_mask = batch['question_mask'].to(device)
                
                # Forward pass in contrastive mode
                outputs = model(images, {'input_ids': question_ids, 'attention_mask': question_mask}, mode='contrastive')
                
                # Compute contrastive loss
                loss = model.contrastive_loss(
                    outputs['image_embeddings'],
                    outputs['text_embeddings']
                )
                
                val_loss += loss.item()
        
        # Print epoch results
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model
            torch.save(model.state_dict(), os.path.join(output_dir, "pretrained_model.pt"))
            print(f"Saved best model with validation loss: {best_loss:.4f}")
    
    print("Pretraining complete!")
    return model

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set parameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 2e-5
    warmup_steps = 1000
    output_dir = "e:/MedVQA/models"
    num_experts = 4
    
    # Run pretraining
    model = pretrain_contrastive(
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        output_dir=output_dir,
        device=device,
        num_experts=num_experts
    )
    
    print("Pretraining finished successfully!")
    return model

if __name__ == "__main__":
    main()