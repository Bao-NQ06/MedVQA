import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from vqa_model import MedicalVQAModel
from dataset import MedicalVQADataset
from cluster_masking import ClusterMasking
from train import calculate_loss

def train_with_masking(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                      num_epochs=10, device='cuda', output_dir='e:/MedVQA/models',
                      mask_ratio=0.5, min_mask_ratio=0.3):
    """Train the VQA model with cluster-based masking.
    
    Args:
        model: MedicalVQAModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Device to train on
        output_dir: Directory to save models
        mask_ratio: Target ratio of patches to mask
        min_mask_ratio: Minimum ratio of patches to mask
    """
    # Initialize cluster masking module
    masking = ClusterMasking(
        patch_size=16,  # Match patch size with vision encoder
        min_mask_ratio=min_mask_ratio,
        distance_threshold=0.5
    ).to(device)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            images = batch['image'].to(device)
            question_ids = batch['question_ids'].to(device)
            question_mask = batch['question_mask'].to(device)
            question_types = batch['is_yes_no'].to(device)
            answers = batch['answer'].to(device)
            
            # Generate masks for images
            with torch.no_grad():
                patch_mask = masking.generate_mask(
                    images,
                    mask_ratio=mask_ratio,
                    use_embedding=False  # Can be set to True if using embeddings
                )
            
            # Apply masking to images (set masked patches to zero)
            N, C, H, W = images.shape
            P = masking.patch_size
            num_patches = (H // P) * (W // P)
            
            # Reshape mask to match image dimensions
            mask_reshaped = patch_mask.reshape(N, H // P, W // P, 1, 1)
            mask_reshaped = mask_reshaped.repeat(1, 1, 1, P, P)
            mask_reshaped = mask_reshaped.permute(0, 3, 1, 4, 2).reshape(N, 1, H, W)
            mask_reshaped = mask_reshaped.repeat(1, C, 1, 1)
            
            # Apply mask
            masked_images = images * (~mask_reshaped)
            
            # Prepare question input
            questions = {'input_ids': question_ids, 'attention_mask': question_mask}
            
            # Forward pass with masked images
            outputs = model(masked_images, questions)
            
            # Calculate loss using the optimized loss function
            loss, loss_components = calculate_loss(outputs, question_types, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update progress bar with detailed loss components
            train_loss += loss.item()
            progress_bar.set_postfix({
                'loss': loss.item(),
                **{f"{k}_loss": v for k, v in loss_components.items() if v > 0}
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_loss_components = {}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                images = batch['image'].to(device)
                question_ids = batch['question_ids'].to(device)
                question_mask = batch['question_mask'].to(device)
                question_types = batch['is_yes_no'].to(device)
                answers = batch['answer'].to(device)
                
                # Generate masks for validation images
                patch_mask = masking.generate_mask(
                    images,
                    mask_ratio=mask_ratio,
                    use_embedding=False
                )
                
                # Apply masking
                N, C, H, W = images.shape
                P = masking.patch_size
                mask_reshaped = patch_mask.reshape(N, H // P, W // P, 1, 1)
                mask_reshaped = mask_reshaped.repeat(1, 1, 1, P, P)
                mask_reshaped = mask_reshaped.permute(0, 3, 1, 4, 2).reshape(N, 1, H, W)
                mask_reshaped = mask_reshaped.repeat(1, C, 1, 1)
                masked_images = images * (~mask_reshaped)
                
                # Prepare question input
                questions = {'input_ids': question_ids, 'attention_mask': question_mask}
                
                # Forward pass
                outputs = model(masked_images, questions)
                
                # Calculate loss using the optimized loss function
                loss, loss_components = calculate_loss(outputs, question_types, batch)
                val_loss += loss.item()
                
                # Update validation loss components
                for k, v in loss_components.items():
                    if k in val_loss_components:
                        val_loss_components[k] += v
        
        # Print epoch results
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model
            torch.save(model.state_dict(), os.path.join(output_dir, "masked_model.pt"))
            print(f"Saved best model with validation loss: {best_loss:.4f}")
    
    print("Training with masking complete!")
    return model

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = MedicalVQAModel().to(device)
    
    # Load datasets
    train_dataset = MedicalVQADataset(split='train')
    val_dataset = MedicalVQADataset(split='val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Train model with masking
    trained_model = train_with_masking(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=10,
        device=device,
        mask_ratio=0.5,  # 50% of patches will be masked
        min_mask_ratio=0.3  # At least 30% of patches will be masked
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()