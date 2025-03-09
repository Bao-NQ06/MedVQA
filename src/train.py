import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os

def train_model(model, train_dataloader, val_dataloader, num_epochs=10, output_dir="e:/MedVQA/models"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Learning rate scheduler
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=total_steps
    )
    
    # Loss functions
    question_type_criterion = nn.CrossEntropyLoss()
    yes_no_criterion = nn.CrossEntropyLoss()
    open_ended_criterion = nn.CrossEntropyLoss()
    
    # Track best validation loss
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            images = batch['image'].to(device)
            question_ids = batch['question_ids'].to(device)
            question_mask = batch['question_mask'].to(device)
            question_types = batch['is_yes_no'].to(device)
            answers = batch['answer'].to(device)
            
            # Prepare question input
            questions = {'input_ids': question_ids, 'attention_mask': question_mask}
            
            # Forward pass
            outputs = model(images, questions)
            
            # Calculate losses
            question_type_loss = question_type_criterion(
                outputs['question_type'], 
                question_types.long()
            )
            
            # Calculate yes/no loss only for yes/no questions
            yes_no_mask = question_types.bool()
            if yes_no_mask.any():
                yes_no_loss = yes_no_criterion(
                    outputs['yes_no_logits'][yes_no_mask],
                    answers[yes_no_mask].long()
                )
            else:
                yes_no_loss = torch.tensor(0.0, device=device)
            
            # Calculate open-ended loss only for open-ended questions
            open_ended_mask = ~yes_no_mask
            if open_ended_mask.any():
                # For open-ended questions, we need to reshape the logits and answers
                open_ended_logits = outputs['open_ended_logits'][open_ended_mask]
                open_ended_answers = answers[open_ended_mask]
                
                # Handle different shapes for MoE output
                if len(open_ended_logits.shape) == 2:
                    # Direct output from MoE
                    open_ended_loss = open_ended_criterion(
                        open_ended_logits,
                        open_ended_answers.long()
                    )
                else:
                    # Sequence output
                    open_ended_loss = open_ended_criterion(
                        open_ended_logits.view(-1, model.biogpt.config.vocab_size),
                        open_ended_answers.view(-1).long()
                    )
            else:
                open_ended_loss = torch.tensor(0.0, device=device)
            
            # Total loss with weighting
            total_loss = question_type_loss + yes_no_loss + open_ended_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            train_loss += total_loss.item()
            progress_bar.set_postfix({'loss': total_loss.item()})
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                # Move batch to device
                images = batch['image'].to(device)
                question_ids = batch['question_ids'].to(device)
                question_mask = batch['question_mask'].to(device)
                question_types = batch['is_yes_no'].to(device)
                answers = batch['answer'].to(device)
                
                # Prepare question input
                questions = {'input_ids': question_ids, 'attention_mask': question_mask}
                
                # Forward pass
                outputs = model(images, questions)
                
                # Calculate losses (same as training)
                question_type_loss = question_type_criterion(
                    outputs['question_type'], 
                    question_types.long()
                )
                
                yes_no_mask = question_types.bool()
                if yes_no_mask.any():
                    yes_no_loss = yes_no_criterion(
                        outputs['yes_no_logits'][yes_no_mask],
                        answers[yes_no_mask].long()
                    )
                else:
                    yes_no_loss = torch.tensor(0.0, device=device)
                
                open_ended_mask = ~yes_no_mask
                if open_ended_mask.any():
                    open_ended_logits = outputs['open_ended_logits'][open_ended_mask]
                    open_ended_answers = answers[open_ended_mask]
                    
                    if len(open_ended_logits.shape) == 2:
                        open_ended_loss = open_ended_criterion(
                            open_ended_logits,
                            open_ended_answers.long()
                        )
                    else:
                        open_ended_loss = open_ended_criterion(
                            open_ended_logits.view(-1, model.biogpt.config.vocab_size),
                            open_ended_answers.view(-1).long()
                        )
                else:
                    open_ended_loss = torch.tensor(0.0, device=device)
                
                total_loss = question_type_loss + yes_no_loss + open_ended_loss
                val_loss += total_loss.item()
        
        # Print epoch results
        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(f"Saved best model with validation loss: {best_loss:.4f}")
    
    print("Training complete!")
    return model