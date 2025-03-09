import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BioGptModel, BlipModel

class ExpertLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.activation(self.linear(x))

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k  # Top-k experts to use
        
        # Create experts
        self.experts = nn.ModuleList([
            ExpertLayer(input_dim, output_dim) for _ in range(num_experts)
        ])
        
        # Router network
        self.router = nn.Linear(input_dim, num_experts)
        
    def forward(self, x):
        # Get routing weights
        routing_weights = F.softmax(self.router(x), dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)  # Normalize weights
        
        # Apply experts
        batch_size = x.size(0)
        expert_outputs = torch.zeros(batch_size, x.size(1), device=x.device)
        
        for i in range(self.k):
            # For each position in top-k
            for j in range(batch_size):
                expert_idx = top_k_indices[j, i]
                expert_output = self.experts[expert_idx](x[j].unsqueeze(0)).squeeze(0)
                expert_outputs[j] += top_k_weights[j, i] * expert_output
                
        return expert_outputs

class KVCache:
    def __init__(self, max_length=100):
        self.max_length = max_length
        self.clear()
        
    def clear(self):
        self.keys = None
        self.values = None
        self.length = 0
        
    def update(self, key, value):
        if self.keys is None:
            self.keys = key
            self.values = value
        else:
            self.keys = torch.cat([self.keys, key], dim=1)
            self.values = torch.cat([self.values, value], dim=1)
            
            # Trim if exceeding max length
            if self.keys.size(1) > self.max_length:
                self.keys = self.keys[:, -self.max_length:]
                self.values = self.values[:, -self.max_length:]
                
        self.length = self.keys.size(1)
        
    def get(self):
        return self.keys, self.values

class MedicalVQAModel(nn.Module):
    def __init__(self, temperature=0.07, num_experts=4, vocab_size=None):
        super().__init__()
        # Load pretrained models
        self.blip = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.biogpt = BioGptModel.from_pretrained("microsoft/biogpt")
        
        if vocab_size is None:
            vocab_size = self.biogpt.config.vocab_size
        
        # Freeze BLIP and BERT weights
        for param in self.blip.parameters():
            param.requires_grad = False
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Question type classifier
        self.question_classifier = nn.Linear(768, 2)  # Binary classification: yes/no vs open-ended
        
        # Yes/No classifier
        self.yes_no_classifier = nn.Linear(768 * 2, 2)
        
        # Projection layers
        self.visual_projection = nn.Linear(768, 768)
        self.question_projection = nn.Linear(768, 768)
        
        # Mixture of Experts for decoder
        self.moe = MixtureOfExperts(768 * 2, 768, num_experts=num_experts)
        
        # Output generation layer
        self.output_projection = nn.Linear(768, vocab_size)
        
        # Contrastive learning components
        self.image_projection_cl = nn.Linear(768, 256)
        self.text_projection_cl = nn.Linear(768, 256)
        self.temperature = temperature
        
        # Anti-repetition mechanism
        self.repetition_penalty = 1.2
        
        # KV Cache for generation
        self.kv_cache = KVCache()
        
        # Medical term emphasis
        self.medical_term_ids = None  # Will be populated with medical term token IDs
        self.medical_boost_factor = 1.3

    def forward(self, image, question, mode='vqa', past_tokens=None):
        # Process image through BLIP's ViT
        vision_outputs = self.blip.vision_model(image)
        vision_features = vision_outputs.last_hidden_state.mean(dim=1)  # Pool visual features
        
        # Process question through BERT
        question_outputs = self.bert(question)
        question_features = question_outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        
        # For contrastive learning mode
        if mode == 'contrastive':
            # Project features to contrastive space
            image_embeddings = self.image_projection_cl(vision_features)
            text_embeddings = self.text_projection_cl(question_features)
            
            # Normalize embeddings
            image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
            
            return {
                'image_embeddings': image_embeddings,
                'text_embeddings': text_embeddings
            }
        
        # For VQA mode (standard forward pass)
        # Project features
        vision_projected = self.visual_projection(vision_features)
        question_projected = self.question_projection(question_features)
        
        # Determine question type
        question_type_logits = self.question_classifier(question_features)
        question_type = torch.argmax(question_type_logits, dim=1)
        
        # Combine visual and question features
        combined_features = torch.cat([vision_projected, question_projected], dim=1)
        
        # Handle yes/no questions
        yes_no_logits = self.yes_no_classifier(combined_features)
        
        # Handle open-ended questions using MoE
        moe_features = self.moe(combined_features)
        
        # Use KV Cache for efficient generation
        if past_tokens is not None:
            # Apply repetition penalty
            logits = self.output_projection(moe_features)
            
            # Penalize recently generated tokens to avoid repetition
            if past_tokens.size(0) > 0:
                for i, token_id in enumerate(past_tokens[-20:]):  # Look at last 20 tokens
                    logits[:, token_id] /= self.repetition_penalty
            
            # Boost medical terms
            if self.medical_term_ids is not None:
                logits[:, self.medical_term_ids] *= self.medical_boost_factor
                
            return {
                'question_type': question_type,
                'yes_no_logits': yes_no_logits,
                'open_ended_logits': logits
            }
        else:
            # Standard generation
            biogpt_outputs = self.biogpt(inputs_embeds=moe_features.unsqueeze(1))
            open_ended_logits = self.output_projection(biogpt_outputs.last_hidden_state)
            
            return {
                'question_type': question_type,
                'yes_no_logits': yes_no_logits,
                'open_ended_logits': open_ended_logits
            }
        
    def contrastive_loss(self, image_embeddings, text_embeddings):
        """
        Compute the contrastive loss between image and text embeddings
        """
        # Compute similarity matrix
        logits = torch.matmul(image_embeddings, text_embeddings.transpose(0, 1)) / self.temperature
        
        # Labels are the diagonal elements (matching pairs)
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # Compute loss in both directions (image->text and text->image)
        image_to_text_loss = F.cross_entropy(logits, labels)
        text_to_image_loss = F.cross_entropy(logits.transpose(0, 1), labels)
        
        # Total loss is the average of both directions
        total_loss = (image_to_text_loss + text_to_image_loss) / 2
        
        return total_loss
        
    def set_medical_term_ids(self, term_ids):
        """Set the token IDs of medical terms to boost during generation"""
        self.medical_term_ids = term_ids