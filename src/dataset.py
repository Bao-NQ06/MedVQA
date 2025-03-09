from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from PIL import Image
import requests
from io import BytesIO

class MedicalVQADataset(Dataset):
    def __init__(self, split='train', max_length=128):
        self.split = split
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load datasets from Hugging Face
        self.pathvqa = load_dataset('flaviagiammarino/path-vqa')[split]
        self.vqarad = load_dataset('flaviagiammarino/vqa-rad')[split]
        
        # Combine datasets
        self.pathvqa_data = self._process_pathvqa()
        self.vqarad_data = self._process_vqarad()
        self.data = self.pathvqa_data + self.vqarad_data

    def _process_pathvqa(self):
        data = []
        for item in self.pathvqa:
            answer_lower = item['answer'].lower()
            is_yes_no = answer_lower in ['yes', 'no']
            
            data.append({
                'image_url': item['image_url'],
                'question': item['question'],
                'answer': item['answer'],
                'is_yes_no': is_yes_no
            })
        return data

    def _process_vqarad(self):
        data = []
        for item in self.vqarad:
            answer_lower = item['answer'].lower()
            is_yes_no = answer_lower in ['yes', 'no']
            
            data.append({
                'image_url': item['image_url'],
                'question': item['question'],
                'answer': item['answer'],
                'is_yes_no': is_yes_no
            })
        return data

    def _load_image_from_url(self, url):
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and preprocess image from URL
        image = self._load_image_from_url(item['image_url'])
        
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
            'image': image,
            'question_ids': question_encoding['input_ids'].squeeze(0),
            'question_mask': question_encoding['attention_mask'].squeeze(0),
            'answer': answer_label,
            'is_yes_no': item['is_yes_no']
        }