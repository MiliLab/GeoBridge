import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import CLIPModel, AutoProcessor
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.*')

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.CLIP = CLIPModel.from_pretrained("clip-vit-large-patch14")
        for param in self.CLIP.named_parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.CLIP.get_image_features(pixel_values=x)
        return x

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.CLIP = CLIPModel.from_pretrained("clip-vit-large-patch14")
        for param in self.CLIP.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        x = self.CLIP.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return x

class GeoBridge(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.drone_encoder = ImageEncoder()   
        self.satellite_encoder = ImageEncoder()   
        self.street_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

    def to(self, device):
        self.drone_encoder.to(device)
        self.satellite_encoder.to(device)
        self.street_encoder.to(device)
        self.text_encoder.to(device)
        return super().to(device)

    def forward(self, drone, satellite, street, text):
        drone_features = self.drone_encoder(drone)    
        drone_features = F.normalize(drone_features, dim=1)
        satellite_features = self.satellite_encoder(satellite)  
        satellite_features = F.normalize(satellite_features, dim=1)
        street_features = self.street_encoder(street)
        street_features = F.normalize(street_features, dim=1)
        text = self.text_encoder(text.input_ids, text.attention_mask)
        text = F.normalize(text, dim=1)
        scale = self.logit_scale.exp()
        return drone_features, satellite_features,street_features,text, scale

def make_model(opt):
    model = GeoBridge(opt)
    if os.path.exists(opt.load_from):
        model.load_params(opt.load_from)
    return model
