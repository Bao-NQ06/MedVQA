from datasets import load_dataset
import torch
from torch.utils.data import Dataset, random_split
from transformers import BertTokenizer
from PIL import Image
import io
# Add this import at the top of the file
import torchvision.transforms as transforms

class MedicalVQADataset(Dataset):
    def __init__(self, split='train', max_length=128, val_ratio=0.1):
        self.split = split
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.val_ratio = val_ratio
        
        # Add image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load PathVQA dataset which has train/validation/test splits
        self.pathvqa = load_dataset('flaviagiammarino/path-vqa')
        
        # Load VQA-RAD dataset which only has train/test splits
        self.vqarad = load_dataset('flaviagiammarino/vqa-rad')
        
        # Process datasets based on split
        if split == 'train':
            self.pathvqa_data = self._process_pathvqa('train')
            # For VQA-RAD, use a portion of train set
            vqarad_train = self._process_vqarad('train')
            # Create train/val split for VQA-RAD
            val_size = int(len(vqarad_train) * self.val_ratio)
            train_size = len(vqarad_train) - val_size
            # Use most of the data for training
            self.vqarad_data = vqarad_train[:train_size]
            
        elif split == 'validation':
            self.pathvqa_data = self._process_pathvqa('validation')
            # For VQA-RAD, use a portion of train set for validation
            vqarad_train = self._process_vqarad('train')
            # Create train/val split for VQA-RAD
            val_size = int(len(vqarad_train) * self.val_ratio)
            train_size = len(vqarad_train) - val_size
            # Use a small portion for validation
            self.vqarad_data = vqarad_train[train_size:]
            
        elif split == 'test':
            self.pathvqa_data = self._process_pathvqa('test')
            self.vqarad_data = self._process_vqarad('test')
        
        # Combine datasets
        self.data = self.pathvqa_data + self.vqarad_data
        print(f"Created {split} dataset with {len(self.data)} samples")

    def _process_pathvqa(self, split):
        data = []
        for item in self.pathvqa[split]:
            answer_lower = item['answer'].lower()
            is_yes_no = answer_lower in ['yes', 'no']
            
            data.append({
                'image': item['image'],
                'question': item['question'],
                'answer': item['answer'],
                'is_yes_no': is_yes_no,
                'dataset_type': 'pathvqa'
            })
        return data

    def _process_vqarad(self, split):
        data = []
        for item in self.vqarad[split]:
            answer_lower = item['answer'].lower()
            is_yes_no = answer_lower in ['yes', 'no']
            
            data.append({
                'image': item['image'],
                'question': item['question'],
                'answer': item['answer'],
                'is_yes_no': is_yes_no,
                'dataset_type': 'vqarad'
            })
        return data

    def _process_image(self, image_data, dataset_type):
        """Process image data from the dataset"""
        # Convert to PIL Image if needed
        if not isinstance(image_data, Image.Image):
            image = Image.open(io.BytesIO(image_data['bytes']))
        else:
            image = image_data
        
        # Handle different image formats
        if dataset_type == 'pathvqa':
            # Convert CMYK to RGB if needed
            if image.mode == 'CMYK':
                image = image.convert('RGB')
            # Handle grayscale images in PathVQA too
            elif image.mode == 'L':
                image = image.convert('RGB')
        elif dataset_type == 'vqarad':
            # Convert grayscale to RGB
            if image.mode == 'L':
                image = image.convert('RGB')
        
        # Ensure all images are RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and preprocess image
        image = self._process_image(item['image'], item['dataset_type'])
        # Apply transformation to convert to tensor
        image_tensor = self.transform(image)
        
        # Tokenize question
        question_encoding = self.tokenizer(
            item['question'],
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare answer
        if item['is_yes_no']:
            answer_label = 1 if item['answer'].lower() == 'yes' else 0
        else:
            answer_encoding = self.tokenizer(
                item['answer'],
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )
            answer_label = answer_encoding['input_ids']
        
        return {
            'image': image_tensor,  # Now returning a tensor instead of PIL Image
            'question_ids': question_encoding['input_ids'].squeeze(0),
            'question_mask': question_encoding['attention_mask'].squeeze(0),
            'answer': answer_label,
            'is_yes_no': item['is_yes_no']
        }