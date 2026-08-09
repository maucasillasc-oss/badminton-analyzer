"""
OPT Transformer - Predicts shuttlecock flying direction from player keypoints.
Architecture matches exactly what was trained in Colab/SageMaker.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import pickle


class CoordinateEmbedding(nn.Module):
    """Embedding for 2-player keypoints."""
    def __init__(self, in_channels=34, emb_size=2048):
        super().__init__()
        h = emb_size // 2
        self.p1 = nn.Sequential(nn.Linear(in_channels, h), nn.Linear(h, h))
        self.p2 = nn.Sequential(nn.Linear(in_channels, h), nn.Linear(h, h))

    def forward(self, x):
        # x: (batch, seq_len, 2, 17, 2) -> (batch, seq_len, 2, 34)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        return torch.cat((self.p1(x.select(2, 0)), self.p2(x.select(2, 1))), 2)


class OptimusPrime(nn.Module):
    """
    OPT Transformer.
    Input: (batch, seq_len, 2, 17, 2) - 2 players, 17 keypoints, (x,y)
    Output: (seq_len, batch, num_tokens) - direction per frame
    Tokens: 0=padding, 1=flying_up, 2=flying_down
    """
    def __init__(self, num_tokens=3, dim_model=2048, num_heads=8,
                 num_encoder_layers=8, dim_feedforward=2048, dropout_p=0.1):
        super().__init__()
        self.emb = CoordinateEmbedding(34, dim_model)
        enc_layer = TransformerEncoderLayer(
            dim_model, num_heads, dim_feedforward, dropout_p, batch_first=False
        )
        self.encoder = TransformerEncoder(enc_layer, num_encoder_layers)
        self.dec1 = nn.Linear(dim_model, dim_model)
        self.dec2 = nn.Linear(dim_model, num_tokens)

    def forward(self, src, src_pad_mask=None):
        src = self.emb(src).permute(1, 0, 2)  # (seq, batch, dim)
        out = self.encoder(src, src_key_padding_mask=src_pad_mask)
        return self.dec2(F.relu(self.dec1(out)))


class OptimusPrimeContainer(object):
    """Container for the trained OPT model."""
    
    def __init__(self, opt_path, scaler_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__setup_model(opt_path)
        self.__setup_scaler(scaler_path)

    def __setup_model(self, opt_path):
        self.model = OptimusPrime(
            num_tokens=3, dim_model=2048, num_heads=8,
            num_encoder_layers=8, dim_feedforward=2048, dropout_p=0.1
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(opt_path, map_location=self.device)
        )
        self.model.eval()
        print(f"  OPT loaded ({sum(p.numel() for p in self.model.parameters()):,} params)")

    def __setup_scaler(self, scaler_path):
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    @torch.no_grad()
    def predict(self, input_sequence):
        """
        Predict shuttle direction for each frame.
        Input: list of keypoint frames, each shape (2, 17, 2)
        Output: tensor of direction indices (0=pad, 1=up, 2=down)
        """
        input_sequence = self.__scale(input_sequence)
        pad_mask = (input_sequence.abs().sum(dim=(2, 3, 4)) == 0)
        pred = self.model(input_sequence, src_pad_mask=pad_mask)
        pred_indices = pred.permute(1, 0, 2).argmax(dim=2).squeeze(0)
        return pred_indices

    def __scale(self, input_sequence):
        """Scale keypoints using the pre-fitted scaler."""
        input_sequence = np.array(input_sequence, dtype=np.float32)
        scaled = []
        for frame in input_sequence:
            flat = frame.reshape(1, -1)
            s = self.scaler.transform(flat).reshape(2, 17, 2)
            scaled.append(s)
        
        tensor = torch.tensor(
            np.array(scaled)
        ).unsqueeze(0).to(torch.float32).to(self.device)
        return tensor
