import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
import os

class KnowledgeDistillation:
    def __init__(self, student_model, temperature=3.0, alpha=0.5):
        """
        Initialize knowledge distillation with a student model and BiomedCLIP as teacher
        
        Args:
            student_model: Your MedicalVQAModel
            temperature: Temperature for softening probability distributions
            alpha: Weight for balancing distillation and task-specific losses
        """
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Load BiomedCLIP as teacher model
        print("Loading BiomedCLIP teacher model...")
        self.teacher_model = AutoModel.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        self.teacher_processor = AutoProcessor.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.teacher_model.eval()
        print("Teacher model loaded successfully")
        
        # Projection layer to align student features with teacher features
        self.student_projection = nn.Linear(768, 512)  # Adjust dimensions as needed
    
    def compute_distillation_loss(self, images, questions):
        """
        Compute knowledge distillation loss between teacher and student
        """
        # Get teacher embeddings
        with torch.no_grad():
            # Process inputs for teacher model
            teacher_inputs = self.teacher_processor(
                text=questions,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(images.device)
            
            # Get teacher embeddings
            teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_image_embeds = teacher_outputs.image_embeds
            teacher_text_embeds = teacher_outputs.text_embeds
        
        # Get student embeddings (in contrastive mode)
        student_outputs = self.student(images, questions, mode='contrastive')
        student_image_embeds = student_outputs['image_embeddings']
        student_text_embeds = student_outputs['text_embeddings']
        
        # Project student embeddings to match teacher dimensions
        student_image_embeds = self.student_projection(student_image_embeds)
        student_text_embeds = self.student_projection(student_text_embeds)
        
        # Normalize embeddings
        teacher_image_embeds = F.normalize(teacher_image_embeds, p=2, dim=1)
        teacher_text_embeds = F.normalize(teacher_text_embeds, p=2, dim=1)
        student_image_embeds = F.normalize(student_image_embeds, p=2, dim=1)
        student_text_embeds = F.normalize(student_text_embeds, p=2, dim=1)
        
        # Compute distillation loss using cosine similarity
        image_distillation_loss = 1 - F.cosine_similarity(student_image_embeds, teacher_image_embeds).mean()
        text_distillation_loss = 1 - F.cosine_similarity(student_text_embeds, teacher_text_embeds).mean()
        
        # Total distillation loss
        distillation_loss = image_distillation_loss + text_distillation_loss
        
        return distillation_loss
    
    def train_with_distillation(self, train_loader, val_loader, task_criterion, 
                               optimizer, scheduler, num_epochs=10, 
                               output_dir="e:/MedVQA/models"):
        """
        Train the student model with knowledge distillation
        """
        device = next(self.student.parameters()).device
        self.teacher_model.to(device)
        self.student_projection.to(device)
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.student.train()
            train_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                # Move batch to device
                images = batch['image'].to(device)
                question_ids = batch['question_ids'].to(device)
                question_mask = batch['question_mask'].to(device)
                question_types = batch['is_yes_no'].to(device)
                answers = batch['answer'].to(device)
                
                # Prepare question input
                questions = {'input_ids': question_ids, 'attention_mask': question_mask}
                
                # Get original questions for teacher model
                original_questions = [batch['question'] for _ in range(len(images))]
                
                # Forward pass for task-specific loss
                outputs = self.student(images, questions)
                
                # Calculate task-specific loss
                task_loss = task_criterion(outputs, question_types, answers)
                
                # Calculate distillation loss
                distillation_loss = self.compute_distillation_loss(images, original_questions)
                
                # Combine losses
                total_loss = (1 - self.alpha) * task_loss + self.alpha * distillation_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Update progress bar
                train_loss += total_loss.item()
                progress_bar.set_postfix({
                    'loss': total_loss.item(),
                    'task_loss': task_loss.item(),
                    'distill_loss': distillation_loss.item()
                })
            
            # Validation
            self.student.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    # Move batch to device
                    images = batch['image'].to(device)
                    question_ids = batch['question_ids'].to(device)
                    question_mask = batch['question_mask'].to(device)
                    question_types = batch['is_yes_no'].to(device)
                    answers = batch['answer'].to(device)
                    
                    # Prepare question input
                    questions = {'input_ids': question_ids, 'attention_mask': question_mask}
                    
                    # Forward pass for task-specific loss
                    outputs = self.student(images, questions)
                    
                    # Calculate task-specific loss
                    task_loss = task_criterion(outputs, question_types, answers)
                    
                    val_loss += task_loss.item()
            
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
                torch.save(self.student.state_dict(), os.path.join(output_dir, "distilled_model.pt"))
                print(f"Saved best model with validation loss: {best_loss:.4f}")
        
        print("Knowledge distillation training complete!")
        return self.student