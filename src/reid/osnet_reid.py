import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np


class OSNetReIDExtractor:
    """
    Deep Person Re-Identification Feature Extractor.
    Extracts L2-normalized 512-d embeddings using PyTorch backbones (OSNet or ResNet-ReID).
    Used for multi-camera person identity matching and cross-video fusion.
    """
    def __init__(self, model_name="osnet_x1_0", device=None, feature_dim=512):
        self.model_name = model_name
        self.feature_dim = feature_dim
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[OSNetReIDExtractor] Initializing ReID backbone ({model_name}) on {self.device}")
        
        # Try importing torchreid first
        self.torchreid_model = None
        try:
            import torchreid
            self.model = torchreid.models.build_model(
                name=model_name,
                num_classes=1000,
                pretrained=True
            )
            self.model.eval()
            self.model.to(self.device)
            self.torchreid_model = True
            print("[OSNetReIDExtractor] Successfully loaded pretrained Torchreid OSNet model.")
        except Exception as e:
            print(f"[OSNetReIDExtractor] Torchreid load notice ({e}). Using PyTorch ResNet50 ReID backbone.")
            # Fallback high quality feature extractor
            resnet = models.resnet50(pretrained=True)
            resnet.fc = nn.Identity()  # 2048-dim features
            self.model = nn.Sequential(
                resnet,
                nn.Linear(2048, feature_dim),
                nn.BatchNorm1d(feature_dim)
            )
            self.model.eval()
            self.model.to(self.device)
            self.torchreid_model = False

        # Standard ReID image preprocessing transform
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract_feature(self, crop_bgr):
        """
        Extract L2-normalized embedding for a single BGR person crop.
        Returns numpy array of shape (512,).
        """
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
            return np.zeros((self.feature_dim,), dtype=np.float32)
            
        crop_rgb = Image.fromarray(crop_bgr[..., ::-1])
        img_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        
        feat = self.model(img_tensor)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
            
        # L2 Normalization
        feat = nn.functional.normalize(feat, p=2, dim=1)
        return feat.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def extract_features_batch(self, crop_bgr_list, batch_size=32):
        """Extract embeddings for a batch of BGR crops."""
        if not crop_bgr_list:
            return np.empty((0, self.feature_dim), dtype=np.float32)
            
        valid_crops = []
        valid_indices = []
        for i, crop in enumerate(crop_bgr_list):
            if crop is not None and crop.size > 0 and crop.shape[0] >= 5 and crop.shape[1] >= 5:
                crop_rgb = Image.fromarray(crop[..., ::-1])
                valid_crops.append(self.transform(crop_rgb))
                valid_indices.append(i)
                
        all_embeddings = np.zeros((len(crop_bgr_list), self.feature_dim), dtype=np.float32)
        if not valid_crops:
            return all_embeddings
            
        tensors = torch.stack(valid_crops).to(self.device)
        num_samples = tensors.shape[0]
        embeddings_list = []
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch = tensors[start:end]
            feats = self.model(batch)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            feats = nn.functional.normalize(feats, p=2, dim=1)
            embeddings_list.append(feats.cpu().numpy())
            
        feats_matrix = np.vstack(embeddings_list)
        for idx, orig_idx in enumerate(valid_indices):
            all_embeddings[orig_idx] = feats_matrix[idx]
            
        return all_embeddings

    @staticmethod
    def compute_cosine_distance(feat1, feat2):
        """
        Compute Cosine Distance between feature embeddings (0 = identical, 1 = orthogonal).
        """
        feat1 = np.asarray(feat1)
        feat2 = np.asarray(feat2)
        
        if feat1.ndim == 1:
            feat1 = feat1.reshape(1, -1)
        if feat2.ndim == 1:
            feat2 = feat2.reshape(1, -1)
            
        # Cosine distance = 1 - cosine similarity
        sim = np.dot(feat1, feat2.T)
        dist = 1.0 - sim
        return np.clip(dist, 0.0, 2.0)
