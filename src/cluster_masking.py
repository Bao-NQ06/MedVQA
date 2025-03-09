import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ClusterMasking(nn.Module):
    def __init__(self, patch_size=16, min_mask_ratio=0.5, distance_threshold=0.5):
        super().__init__()
        self.patch_size = patch_size
        self.min_mask_ratio = min_mask_ratio
        self.distance_threshold = distance_threshold
        
    def patchify(self, images):
        """Split images into patches.
        
        Args:
            images (torch.Tensor): Input images of shape (N, C, H, W)
            
        Returns:
            torch.Tensor: Patches of shape (N, L, patch_size*patch_size*C)
            where L is the number of patches
        """
        N, C, H, W = images.shape
        p = self.patch_size
        
        # Ensure image dimensions are divisible by patch size
        assert H % p == 0 and W % p == 0, f'Image dimensions must be divisible by patch size {p}'
        
        # Reshape and permute to get patches
        patches = images.reshape(N, C, H//p, p, W//p, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(N, -1, p*p*C)
        return patches
    
    def normalize_patches(self, patches):
        """Normalize patches by subtracting mean and dividing by std.
        
        Args:
            patches (torch.Tensor): Input patches of shape (N, L, D)
            
        Returns:
            torch.Tensor: Normalized patches
        """
        # Calculate mean and std along feature dimension
        mean = patches.mean(dim=-1, keepdim=True)
        std = patches.std(dim=-1, keepdim=True) + 1e-6
        return (patches - mean) / std
    
    def compute_cosine_similarity(self, patches):
        """Compute pairwise cosine similarity between patches.
        
        Args:
            patches (torch.Tensor): Normalized patches of shape (N, L, D)
            
        Returns:
            torch.Tensor: Similarity matrix of shape (N, L, L)
        """
        # Normalize patch vectors for cosine similarity
        patches_norm = F.normalize(patches, p=2, dim=-1)
        # Compute similarity matrix efficiently using batch matrix multiplication
        similarity = torch.bmm(patches_norm, patches_norm.transpose(1, 2))
        return similarity
    
    def select_anchors(self, num_patches, mask_ratio, device):
        """Randomly select anchor patches.
        
        Args:
            num_patches (int): Total number of patches
            mask_ratio (float): Target mask ratio
            device: Device to create tensor on
            
        Returns:
            torch.Tensor: Boolean tensor indicating anchor patches
        """
        num_anchors = max(1, int(num_patches * mask_ratio * 0.05))  # 5% of target masked patches
        anchor_indices = torch.randperm(num_patches, device=device)[:num_anchors]
        anchors = torch.zeros(num_patches, dtype=torch.bool, device=device)
        anchors[anchor_indices] = True
        return anchors
    
    def adjust_threshold(self, similarity, anchors, target_ratio, steps=10):
        """Binary search for threshold to achieve target mask ratio.
        
        Args:
            similarity (torch.Tensor): Similarity matrix (N, L, L)
            anchors (torch.Tensor): Anchor patch indicators (N, L)
            target_ratio (float): Target mask ratio
            steps (int): Number of binary search steps
            
        Returns:
            float: Adjusted threshold
        """
        left, right = 0.0, 1.0
        best_threshold = self.distance_threshold
        best_ratio_diff = float('inf')
        
        for _ in range(steps):
            threshold = (left + right) / 2
            # Get patches above threshold for each anchor
            mask = (similarity > threshold).any(dim=-1)
            current_ratio = mask.float().mean().item()
            
            ratio_diff = abs(current_ratio - target_ratio)
            if ratio_diff < best_ratio_diff:
                best_threshold = threshold
                best_ratio_diff = ratio_diff
            
            if current_ratio > target_ratio:
                left = threshold
            else:
                right = threshold
                
        return best_threshold
    
    def generate_mask(self, img, mask_ratio, use_embedding=False, alpha=0.0, embeddings=None):
        """Generate cluster-based mask for input images.
        
        Args:
            img (torch.Tensor): Input images (N, C, H, W)
            mask_ratio (float): Target mask ratio
            use_embedding (bool): Whether to use embedding features
            alpha (float): Weight for RGB vs embedding similarity
            embeddings (torch.Tensor, optional): Pre-computed patch embeddings
            
        Returns:
            torch.Tensor: Boolean mask indicating masked patches
        """
        device = img.device
        
        # Extract and normalize patches
        patches = self.patchify(img)
        norm_patches = self.normalize_patches(patches)
        
        # Compute RGB-based similarity
        rgb_similarity = self.compute_cosine_similarity(norm_patches)
        
        if use_embedding and embeddings is not None:
            # Compute embedding-based similarity
            emb_similarity = self.compute_cosine_similarity(embeddings)
            # Combine similarities with weight alpha
            similarity = alpha * rgb_similarity + (1 - alpha) * emb_similarity
        else:
            similarity = rgb_similarity
        
        # Select anchor patches
        anchors = self.select_anchors(patches.size(1), mask_ratio, device)
        
        # Adjust threshold to meet target ratio
        threshold = self.adjust_threshold(similarity, anchors, mask_ratio)
        
        # Generate initial mask based on similarity to anchors
        anchor_similarity = similarity[:, anchors]
        mask = (anchor_similarity > threshold).any(dim=1)
        
        # Ensure minimum mask ratio is met
        current_ratio = mask.float().mean(dim=1)
        for i in range(mask.size(0)):
            if current_ratio[i] < self.min_mask_ratio:
                # Randomly select additional patches to mask
                n_additional = int((self.min_mask_ratio - current_ratio[i]) * mask.size(1))
                unmasked = (~mask[i]).nonzero().squeeze()
                if unmasked.numel() > 0:  # Check if there are unmasked patches
                    additional = unmasked[torch.randperm(unmasked.numel())[:n_additional]]
                    mask[i][additional] = True
        
        return mask