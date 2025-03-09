import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
from vqa_model import MedicalVQAModel
from dataset import MedicalVQADataset
from knowledge_distillation import KnowledgeDistillation

def task_criterion(outputs, question_types, answers):
    """Combined loss function for VQA task"""
    device = outputs['question_type'].device
    
    # Loss functions
    question_type_criterion = nn.CrossEntropyLoss()
    yes_no_criterion = nn.CrossEntropyLoss()
    open_ended_criterion = nn.CrossEntropyLoss()
    
    # Calculate question type loss
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
            vocab_size = open_ended_logits.size(-1)
            open_ended_loss = open_ended_criterion(
                open_ended_logits.view(-1, vocab_size),
                open_ended_answers.view(-1).long()
            )
    else:
        open_ended_loss = torch.tensor(0.0, device=device)
    
    # Total loss
    total_loss = question_type_loss + yes_no_loss + open_ended_loss
    
    return total_loss

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = MedicalVQAModel()
    
    # Load pretrained weights if available
    pretrained_path = "e:/MedVQA/models/pretrained_model.pt"
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    
    model.to(device)
    
    # Create datasets
    train_dataset = MedicalVQADataset(split='train')
    val_dataset = MedicalVQADataset(split='validation')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,  # Smaller batch size due to memory constraints with two models
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )
    
    # Setup knowledge distillation
    distiller = KnowledgeDistillation(
        student_model=model,
        temperature=3.0,
        alpha=0.5  # Balance between distillation and task-specific losses
    )
    
    # Optimizer
    optimizer = AdamW(
        list(model.parameters()) + list(distiller.student_projection.parameters()),
        lr=2e-5
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * 10  # 10 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=total_steps
    )
    
    # Train with knowledge distillation
    distiller.train_with_distillation(
        train_loader=train_loader,
        val_loader=val_loader,
        task_criterion=task_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=10,
        output_dir="e:/MedVQA/models"
    )
    
    print("Training with knowledge distillation complete!")

if __name__ == "__main__":
    main()