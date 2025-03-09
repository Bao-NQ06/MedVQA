import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import nltk
from transformers import BertTokenizer

from vqa_model import MedicalVQAModel
from dataset import MedicalVQADataset
from train import train_model

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Custom collate function to handle mixed data types
def custom_collate_fn(batch):
    # Filter out any None values
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
    
    # Create a batch dictionary with only essential data
    # Convert images to half precision (float16) to save memory
    batch_dict = {
        'image': torch.stack([item['image'] for item in batch]).to(torch.float16),
        'question_ids': torch.stack([item['question_ids'] for item in batch]),
        'question_mask': torch.stack([item['question_mask'] for item in batch]),
        'is_yes_no': torch.tensor([item['is_yes_no'] for item in batch]),
    }
    
    # Handle answers based on question type
    yes_no_indices = []
    open_ended_indices = []
    
    for i, item in enumerate(batch):
        if item['is_yes_no']:
            yes_no_indices.append(i)
        else:
            open_ended_indices.append(i)
    
    # Process yes/no and open-ended answers separately
    if yes_no_indices:
        yes_no_answers = [batch[i]['answer'] for i in yes_no_indices]
        batch_dict['yes_no_answers'] = torch.tensor(yes_no_answers)
        batch_dict['yes_no_indices'] = torch.tensor(yes_no_indices)
    
    if open_ended_indices:
        open_ended_answers = [batch[i]['answer'] for i in open_ended_indices]
        # Handle tensor answers
        if all(isinstance(ans, torch.Tensor) for ans in open_ended_answers):
            # Pad to same length if needed
            max_len = max(ans.size(0) for ans in open_ended_answers)
            padded_answers = []
            for ans in open_ended_answers:
                if ans.size(0) < max_len:
                    padding = torch.zeros(max_len - ans.size(0), dtype=ans.dtype, device=ans.device)
                    padded_ans = torch.cat([ans, padding])
                else:
                    padded_ans = ans
                padded_answers.append(padded_ans)
            batch_dict['open_ended_answers'] = torch.stack(padded_answers)
        else:
            batch_dict['open_ended_answers'] = open_ended_answers
        batch_dict['open_ended_indices'] = torch.tensor(open_ended_indices)
    
    # Remove the redundant all_answers processing to save memory
    return batch_dict

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enable memory optimization
    torch.cuda.empty_cache()
    
    # Enable deterministic algorithms for better memory usage
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set smaller batch size to reduce memory usage
    batch_size = 4  # Reduced from 8
    print(f"Creating data loaders with batch size {batch_size}...")
    
    # Use gradient accumulation to simulate larger batch sizes
    gradient_accumulation_steps = 4  # Increased from 2
    effective_batch_size = batch_size * gradient_accumulation_steps
    print(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
    print(f"Effective batch size: {effective_batch_size}")
    
    # Create datasets with proper handling for VQA-RAD validation
    print("Loading datasets...")
    train_dataset = MedicalVQADataset(split='train', val_ratio=0.1)
    val_dataset = MedicalVQADataset(split='validation', val_ratio=0.1)
    test_dataset = MedicalVQADataset(split='test')
    
    # Create data loaders with custom collate function and no workers to save memory
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # No multiprocessing to save memory
        collate_fn=custom_collate_fn,
        pin_memory=True  # Faster data transfer to GPU
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # No multiprocessing to save memory
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # No multiprocessing to save memory
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # Create model with memory optimization
    print("Initializing model...")
    model = MedicalVQAModel()
    model.to(device)  # Move model to device immediately after creation
    
    # Set output directory
    output_dir = "e:\\MedVQA\\models"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if model already exists
    model_path = os.path.join(output_dir, "best_model.pt")
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        # Load model with map_location to avoid memory spikes
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Training new model...")
        # Train model with memory optimizations
        model = train_model(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=10,
            output_dir=output_dir,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
    
    # Test model
    print("Testing model...")
    test_results = test_model(model, test_loader, device)
    
    # Print results
    print("\n===== Test Results =====")
    if 'yes_no_accuracy' in test_results:
        print(f"Yes/No Question Accuracy: {test_results['yes_no_accuracy']:.4f}")
        print(f"Yes/No Question F1 Score: {test_results['yes_no_f1']:.4f}")
    else:
        print("No yes/no questions in test set")
    
    if 'open_ended_bleu' in test_results:
        print(f"Open-Ended Question BLEU-1 Score: {test_results['open_ended_bleu']:.4f}")
    else:
        print("No open-ended questions in test set or BLEU calculation failed")
    
    if 'open_ended_rouge' in test_results:
        print(f"Open-Ended Question ROUGE-1 Score: {test_results['open_ended_rouge']:.4f}")
    else:
        print("No open-ended questions in test set or ROUGE calculation failed")

def test_model(model, test_loader, device):
    """
    Test the model on the test dataset
    
    Args:
        model: The MedicalVQAModel to test
        test_loader: DataLoader for the test dataset
        device: Device to run the model on
        
    Returns:
        Dictionary with test results
    """
    model.to(device)
    model.eval()
    
    # Initialize result tracking
    yes_no_preds = []
    yes_no_labels = []
    open_ended_preds = []
    open_ended_labels = []
    
    # Tokenizer for decoding predictions
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Process batches
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        
        for batch_idx, batch in enumerate(progress_bar):
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
                
                # Use mixed precision for forward pass
                with torch.amp.autocast('cuda'):
                    # Forward pass
                    outputs = model(
                        images, 
                        {'input_ids': question_ids, 'attention_mask': question_mask}
                    )
                
                # Process yes/no questions
                if 'yes_no_indices' in batch and len(batch['yes_no_indices']) > 0 and 'yes_no_logits' in outputs:
                    # Get predictions for yes/no questions
                    yes_no_logits = outputs['yes_no_logits'][yes_no_indices]
                    
                    # Handle different output shapes
                    if yes_no_logits.size(-1) == 2:  # Binary classification with 2 outputs
                        batch_yes_no_preds = torch.argmax(yes_no_logits, dim=1).cpu().numpy()
                    else:  # Single output
                        batch_yes_no_preds = (torch.sigmoid(yes_no_logits) > 0.5).int().cpu().numpy()
                    
                    # Store predictions and labels
                    yes_no_preds.extend(batch_yes_no_preds)
                    yes_no_labels.extend(yes_no_answers.cpu().numpy())
                
                # Process open-ended questions
                if 'open_ended_indices' in batch and len(batch['open_ended_indices']) > 0 and 'open_ended_logits' in outputs:
                    # Get predictions for open-ended questions
                    open_ended_logits = outputs['open_ended_logits'][open_ended_indices]
                    
                    # Handle different output shapes
                    if len(open_ended_logits.shape) == 2:  # [batch_size, vocab_size]
                        batch_open_ended_preds = torch.argmax(open_ended_logits, dim=1).unsqueeze(1).cpu().numpy()
                    else:  # [batch_size, seq_len, vocab_size]
                        batch_open_ended_preds = torch.argmax(open_ended_logits, dim=-1).cpu().numpy()
                    
                    # Convert token IDs to text
                    for i, pred_ids in enumerate(batch_open_ended_preds):
                        # Filter out padding and special tokens
                        pred_ids = [id for id in pred_ids if id > 0]
                        if not pred_ids:
                            pred_text = ""
                        else:
                            # Convert prediction to text
                            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                        
                        open_ended_preds.append(pred_text)
                        
                        # Get ground truth answer
                        if isinstance(batch['open_ended_answers'], list):
                            # Text answers
                            open_ended_labels.append(batch['open_ended_answers'][i])
                        else:
                            # Token ID answers
                            answer_ids = batch['open_ended_answers'][i].cpu().numpy()
                            # Filter out padding and special tokens
                            answer_ids = [id for id in answer_ids if id > 0]
                            if not answer_ids:
                                answer_text = ""
                            else:
                                answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
                            open_ended_labels.append(answer_text)
                
                # Display progress
                progress_bar.set_postfix({
                    'yes_no_count': len(yes_no_preds),
                    'open_ended_count': len(open_ended_preds)
                })
                
                # Free up memory
                del images, question_ids, question_mask, outputs
                if 'yes_no_answers' in locals():
                    del yes_no_answers, yes_no_indices
                if 'open_ended_answers' in locals():
                    del open_ended_answers, open_ended_indices
                
                # Force garbage collection
                if batch_idx % 10 == 0:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error in test batch {batch_idx}: {e}")
                continue
    
    # Calculate metrics
    results = {}
    
    # Yes/No metrics
    if yes_no_preds:
        yes_no_preds = np.array(yes_no_preds)
        yes_no_labels = np.array(yes_no_labels)
        
        results['yes_no_accuracy'] = accuracy_score(yes_no_labels, yes_no_preds)
        results['yes_no_f1'] = f1_score(yes_no_labels, yes_no_preds, average='weighted')
    
    # Open-ended metrics
    if open_ended_preds:
        # Calculate BLEU score
        try:
            bleu_scores = []
            for pred, label in zip(open_ended_preds, open_ended_labels):
                # Skip empty predictions or references
                if not pred or not label:
                    continue
                    
                # Tokenize prediction and reference
                pred_tokens = nltk.word_tokenize(pred.lower())
                label_tokens = nltk.word_tokenize(label.lower())
                
                if not pred_tokens or not label_tokens:
                    continue
                
                # Calculate BLEU-1 score
                bleu_score = sentence_bleu([label_tokens], pred_tokens, weights=(1, 0, 0, 0))
                bleu_scores.append(bleu_score)
            
            if bleu_scores:
                results['open_ended_bleu'] = np.mean(bleu_scores)
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
        
        # Calculate ROUGE score
        try:
            rouge = Rouge()
            rouge_scores = []
            
            for pred, label in zip(open_ended_preds, open_ended_labels):
                # Skip empty predictions or references
                if not pred or not label:
                    continue
                    
                # Calculate ROUGE score
                try:
                    rouge_score = rouge.get_scores(pred, label)[0]['rouge-1']['f']
                    rouge_scores.append(rouge_score)
                except Exception as e:
                    # Skip this pair if ROUGE calculation fails
                    print(f"Error calculating ROUGE for a specific pair: {e}")
                    continue
            
            if rouge_scores:
                results['open_ended_rouge'] = np.mean(rouge_scores)
        except Exception as e:
            print(f"Error calculating ROUGE score: {e}")
    
    # Save predictions for visualization
    if open_ended_preds:
        results['open_ended_predictions'] = list(zip(open_ended_preds, open_ended_labels))
    if yes_no_preds:
        results['yes_no_predictions'] = list(zip(yes_no_preds, yes_no_labels))
    
    return results


def visualize_predictions(test_results, num_examples=10, output_dir="e:\\MedVQA\\results"):
    """
    Visualize the model's predictions compared to ground truth
    
    Args:
        test_results: Dictionary with test results including predictions
        num_examples: Number of examples to visualize (default: 10)
        output_dir: Directory to save visualization results
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from tabulate import tabulate
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize open-ended predictions
    if 'open_ended_predictions' in test_results:
        predictions = test_results['open_ended_predictions']
        
        # Limit to specified number of examples
        if len(predictions) > num_examples:
            # Get a mix of good and bad predictions
            predictions = predictions[:num_examples]
        
        # Create DataFrame for better visualization
        df = pd.DataFrame(predictions, columns=['Prediction', 'Ground Truth'])
        
        # Add similarity score
        def simple_similarity(pred, truth):
            # Convert to lowercase and split into words
            pred_words = set(pred.lower().split())
            truth_words = set(truth.lower().split())
            
            # Calculate Jaccard similarity
            if not pred_words or not truth_words:
                return 0.0
            
            intersection = len(pred_words.intersection(truth_words))
            union = len(pred_words.union(truth_words))
            return intersection / union if union > 0 else 0.0
        
        df['Similarity'] = df.apply(lambda row: simple_similarity(row['Prediction'], row['Ground Truth']), axis=1)
        
        # Sort by similarity for better visualization
        df = df.sort_values('Similarity', ascending=False)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, 'open_ended_predictions.csv')
        df.to_csv(csv_path, index=False)
        print(f"Open-ended predictions saved to {csv_path}")
        
        # Print examples in console
        print("\n===== Open-Ended Predictions =====")
        print(tabulate(df.head(num_examples), headers='keys', tablefmt='grid'))
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(df)), df['Similarity'], color='skyblue')
        plt.xlabel('Example Index')
        plt.ylabel('Text Similarity')
        plt.title('Open-Ended Prediction Similarity to Ground Truth')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'open_ended_similarity.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Similarity plot saved to {plot_path}")
    
    # Visualize yes/no predictions
    if 'yes_no_predictions' in test_results:
        predictions = test_results['yes_no_predictions']
        
        # Create DataFrame
        df = pd.DataFrame(predictions, columns=['Prediction', 'Ground Truth'])
        
        # Add correctness column
        df['Correct'] = df['Prediction'] == df['Ground Truth']
        
        # Calculate accuracy
        accuracy = df['Correct'].mean()
        
        # Save to CSV
        csv_path = os.path.join(output_dir, 'yes_no_predictions.csv')
        df.to_csv(csv_path, index=False)
        print(f"Yes/No predictions saved to {csv_path}")
        
        # Print examples in console
        print("\n===== Yes/No Predictions =====")
        print(tabulate(df.head(num_examples), headers='keys', tablefmt='grid'))
        
        # Create visualization
        plt.figure(figsize=(8, 6))
        counts = df['Correct'].value_counts()
        plt.pie(counts, labels=['Incorrect', 'Correct'] if False in counts.index else ['Correct'],
                autopct='%1.1f%%', colors=['salmon', 'lightgreen'])
        plt.title(f'Yes/No Prediction Accuracy: {accuracy:.2%}')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'yes_no_accuracy.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Accuracy plot saved to {plot_path}")


def plot_training_history(training_history, output_dir="e:\\MedVQA\\results"):
    """
    Plot the training and validation loss over epochs
    
    Args:
        training_history: Dictionary with training history
        output_dir: Directory to save visualization results
    """
    import matplotlib.pyplot as plt
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(12, 8))
    plt.plot(training_history['train_loss'], label='Train Loss', marker='o')
    plt.plot(training_history['val_loss'], label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Training history plot saved to {plot_path}")
    
    # Plot component losses if available
    if 'train_components' in training_history and training_history['train_components']:
        plt.figure(figsize=(12, 8))
        
        # Plot each component
        for component in training_history['train_components'][0].keys():
            train_values = [epoch_data[component] for epoch_data in training_history['train_components']]
            val_values = [epoch_data[component] for epoch_data in training_history['val_components']]
            
            plt.plot(train_values, label=f'Train {component}', marker='o')
            plt.plot(val_values, label=f'Val {component}', marker='x')
        
        plt.xlabel('Epoch')
        plt.ylabel('Component Loss')
        plt.title('Training and Validation Component Losses')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'component_losses.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Component losses plot saved to {plot_path}")

