import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os

def train_model(model, train_dataloader, val_dataloader, num_epochs=10, output_dir="e:\\MedVQA\\models", 
                gradient_accumulation_steps=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define loss criteria
    question_type_criterion = nn.CrossEntropyLoss()
    yes_no_criterion = nn.CrossEntropyLoss()
    open_ended_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    # Enable gradient checkpointing in all supported models
    if hasattr(model.blip, 'gradient_checkpointing_enable'):
        model.blip.gradient_checkpointing_enable()
    if hasattr(model.bert, 'gradient_checkpointing_enable'):
        model.bert.gradient_checkpointing_enable()
    if hasattr(model.biogpt, 'gradient_checkpointing_enable'):
        model.biogpt.gradient_checkpointing_enable()
    
    # Move model to device after enabling gradient checkpointing
    model.to(device)
    
    # Use mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Optimizer with weight decay for regularization
    # Use parameter groups to apply different learning rates
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5)
    
    # Learning rate scheduler with warmup
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_loss_components = {'question_type': 0.0, 'yes_no': 0.0, 'open_ended': 0.0}
        
        # Reset gradients at the beginning of each epoch
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Skip empty batches
            if not batch:
                continue
            
            # Clear CUDA cache periodically to prevent memory fragmentation
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                
            try:
                # Move batch to device
                images = batch['image'].to(device, non_blocking=True)
                question_ids = batch['question_ids'].to(device, non_blocking=True)
                question_mask = batch['question_mask'].to(device, non_blocking=True)
                is_yes_no = batch['is_yes_no'].to(device, non_blocking=True)
                
                # Handle yes/no and open-ended answers separately
                if 'yes_no_answers' in batch:
                    yes_no_answers = batch['yes_no_answers'].to(device, non_blocking=True)
                    yes_no_indices = batch['yes_no_indices'].to(device, non_blocking=True)
                
                if 'open_ended_answers' in batch:
                    if isinstance(batch['open_ended_answers'], torch.Tensor):
                        open_ended_answers = batch['open_ended_answers'].to(device, non_blocking=True)
                    else:
                        open_ended_answers = batch['open_ended_answers']
                    open_ended_indices = batch['open_ended_indices'].to(device, non_blocking=True)
                
                # Use mixed precision for forward pass with updated syntax
                with torch.amp.autocast('cuda'):
                    # Forward pass
                    outputs = model(
                        images, 
                        {'input_ids': question_ids, 'attention_mask': question_mask}
                    )
                    
                    # Calculate loss with improved function
                    loss, loss_components = calculate_loss(outputs, is_yes_no, batch)
                
                # Scale loss and backpropagate with gradient accumulation
                scaled_loss = scaler.scale(loss / gradient_accumulation_steps)
                scaled_loss.backward()
                
                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Unscale gradients for gradient clipping
                    scaler.unscale_(optimizer)
                    
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update weights
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Update progress bar
                train_loss += loss.item()
                for k, v in loss_components.items():
                    if k in train_loss_components:
                        train_loss_components[k] += v
                
                # Display current loss components
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    **{f"{k}_loss": v for k, v in loss_components.items() if v > 0}
                })
                
                # Free up memory
                del images, question_ids, question_mask, outputs, loss
                if 'yes_no_answers' in locals():
                    del yes_no_answers, yes_no_indices
                if 'open_ended_answers' in locals():
                    del open_ended_answers, open_ended_indices
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_loss_components = {'question_type': 0.0, 'yes_no': 0.0, 'open_ended': 0.0}
        
        with torch.no_grad():
            progress_bar_val = tqdm(val_dataloader, desc="Validation")
            
            for batch_idx, batch in enumerate(progress_bar_val):
                # Skip empty batches
                if not batch:
                    continue
                
                # Clear CUDA cache periodically to prevent memory fragmentation
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                    
                try:
                    # Move batch to device with non_blocking=True for better performance
                    images = batch['image'].to(device, non_blocking=True)
                    question_ids = batch['question_ids'].to(device, non_blocking=True)
                    question_mask = batch['question_mask'].to(device, non_blocking=True)
                    is_yes_no = batch['is_yes_no'].to(device, non_blocking=True)
                    
                    # Handle yes/no and open-ended answers separately
                    if 'yes_no_answers' in batch:
                        yes_no_answers = batch['yes_no_answers'].to(device, non_blocking=True)
                        yes_no_indices = batch['yes_no_indices'].to(device, non_blocking=True)
                    
                    if 'open_ended_answers' in batch:
                        if isinstance(batch['open_ended_answers'], torch.Tensor):
                            open_ended_answers = batch['open_ended_answers'].to(device, non_blocking=True)
                        else:
                            open_ended_answers = batch['open_ended_answers']
                        open_ended_indices = batch['open_ended_indices'].to(device, non_blocking=True)
                    
                    # Use mixed precision for forward pass with updated syntax
                    with torch.amp.autocast('cuda'):
                        # Forward pass
                        outputs = model(
                            images, 
                            {'input_ids': question_ids, 'attention_mask': question_mask}
                        )
                        
                        # Calculate validation loss with improved function
                        loss, loss_components = calculate_loss(outputs, is_yes_no, batch)
                    
                    val_loss += loss.item()
                    for k, v in loss_components.items():
                        if k in val_loss_components:
                            val_loss_components[k] += v
                    
                    # Display current loss components
                    progress_bar_val.set_postfix({
                        'val_loss': loss.item(),
                        **{f"val_{k}_loss": v for k, v in loss_components.items() if v > 0}
                    })
                    
                    # Free up memory
                    del images, question_ids, question_mask, outputs, loss
                    if 'yes_no_answers' in locals():
                        del yes_no_answers, yes_no_indices
                    if 'open_ended_answers' in locals():
                        del open_ended_answers, open_ended_indices
                            
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear CUDA cache periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        # Print epoch results
        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        
        # Calculate average loss components
        avg_train_components = {k: v / len(train_dataloader) for k, v in train_loss_components.items() if v > 0}
        avg_val_components = {k: v / len(val_dataloader) for k, v in val_loss_components.items() if v > 0}
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Train Loss Components: {avg_train_components}")
        print(f"Val Loss Components: {avg_val_components}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    
    print("Training complete!")
    return model


def calculate_loss(outputs, is_yes_no, batch):
    """Calculate the loss based on model outputs and batch data with improved shape handling"""
    device = is_yes_no.device
    total_loss = torch.tensor(0.0, device=device)
    loss_components = {}
    
    # Question type classification loss (if available in outputs)
    if 'question_type' in outputs:
        # Ensure both inputs are float type
        logits = outputs['question_type'].float()
        
        # Check dimensions and adjust accordingly
        if logits.size(-1) == 2:  # Binary classification with 2 outputs
            # Create one-hot encoding for binary classification
            targets = torch.zeros_like(logits)
            targets[:, 0] = 1.0 - is_yes_no.float()  # Not yes/no
            targets[:, 1] = is_yes_no.float()        # Is yes/no
            
            # Use binary_cross_entropy_with_logits which is safe for autocast
            question_type_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,  # Use raw logits (float)
                targets  # Target as one-hot encoding
            )
        else:  # Single output
            # Use binary_cross_entropy_with_logits for single output
            question_type_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.view(-1),  # Flatten to match target
                is_yes_no.float()  # Target as float
            )
            
        total_loss += question_type_loss
        loss_components['question_type'] = question_type_loss.item()
    
    # Handle yes/no questions
    if 'yes_no_indices' in batch and len(batch['yes_no_indices']) > 0:
        yes_no_indices = batch['yes_no_indices'].to(device)
        yes_no_answers = batch['yes_no_answers'].to(device)
        
        if len(yes_no_indices) > 0 and 'yes_no_logits' in outputs:
            # Get yes/no logits for the corresponding indices only
            yes_no_logits = outputs['yes_no_logits'][yes_no_indices].float()
            yes_no_targets = yes_no_answers.float()
            
            # Check dimensions and adjust accordingly
            if yes_no_logits.size(-1) == 2:  # Binary classification with 2 outputs
                # Create one-hot encoding for binary classification
                targets = torch.zeros_like(yes_no_logits)
                targets[:, 0] = 1.0 - yes_no_targets  # No
                targets[:, 1] = yes_no_targets        # Yes
                
                # Use binary_cross_entropy_with_logits which is safe for autocast
                yes_no_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    yes_no_logits,  # Use raw logits (float)
                    targets  # Target as one-hot encoding
                )
            else:  # Single output
                # Use binary_cross_entropy_with_logits for single output
                yes_no_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    yes_no_logits.view(-1),  # Flatten to match target
                    yes_no_targets  # Target as float
                )
                
            total_loss += yes_no_loss
            loss_components['yes_no'] = yes_no_loss.item()
    
    # Handle open-ended questions
    if 'open_ended_indices' in batch and len(batch['open_ended_indices']) > 0:
        open_ended_indices = batch['open_ended_indices'].to(device)
        
        if len(open_ended_indices) > 0 and 'open_ended_logits' in outputs:
            # Get open-ended logits for the corresponding indices only
            open_ended_logits = outputs['open_ended_logits'][open_ended_indices]
            
            if isinstance(batch['open_ended_answers'], list):
                # Text answers - skip loss calculation for now
                pass
            else:
                # Token ID answers
                open_ended_answers = batch['open_ended_answers'].to(device)
                
                # Handle different output shapes
                if len(open_ended_logits.shape) == 2:  # [batch_size, vocab_size]
                    # For single token prediction
                    if open_ended_logits.size(0) == open_ended_answers.size(0):
                        # Use cross entropy loss which is safe for autocast
                        open_ended_loss = torch.nn.functional.cross_entropy(
                            open_ended_logits.float(),  # Logits as float
                            open_ended_answers.long(),  # Target as long for cross_entropy
                            ignore_index=0
                        )
                        total_loss += open_ended_loss
                        loss_components['open_ended'] = open_ended_loss.item()
                
                elif len(open_ended_logits.shape) == 3:  # [batch_size, seq_len, vocab_size]
                    batch_size, seq_len, vocab_size = open_ended_logits.shape
                    
                    # Ensure answers has the right shape for sequence prediction
                    if open_ended_answers.dim() == 1:
                        # If answers is 1D but we need 2D, reshape it
                        if open_ended_answers.size(0) == batch_size:
                            # Single answer per example - use only first token prediction
                            # Use cross entropy loss which is safe for autocast
                            open_ended_loss = torch.nn.functional.cross_entropy(
                                open_ended_logits[:, 0, :].float(),  # Use only first position
                                open_ended_answers.long(),   # Target as long for cross_entropy
                                ignore_index=0
                            )
                            total_loss += open_ended_loss
                            loss_components['open_ended'] = open_ended_loss.item()
                    else:
                        # Reshape for sequence prediction
                        flat_logits = open_ended_logits.reshape(-1, vocab_size).float()
                        flat_answers = open_ended_answers.reshape(-1).long()
                        
                        # Check if shapes are compatible after flattening
                        if flat_logits.size(0) == flat_answers.size(0):
                            # Use cross entropy loss which is safe for autocast
                            open_ended_loss = torch.nn.functional.cross_entropy(
                                flat_logits,
                                flat_answers,
                                ignore_index=0
                            )
                            total_loss += open_ended_loss
                            loss_components['open_ended'] = open_ended_loss.item()
                        else:
                            # Use truncation to match sizes
                            min_size = min(flat_logits.size(0), flat_answers.size(0))
                            
                            # Use cross entropy loss which is safe for autocast
                            open_ended_loss = torch.nn.functional.cross_entropy(
                                flat_logits[:min_size],
                                flat_answers[:min_size],
                                ignore_index=0
                            )
                            total_loss += open_ended_loss
                            loss_components['open_ended'] = open_ended_loss.item()
    
    return total_loss, loss_components