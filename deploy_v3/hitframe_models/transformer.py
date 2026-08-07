"""
OPT (Optimus Prime Transformer): Predicts shuttlecock flying direction 
from player keypoint sequences.
Output: direction per frame (0=no movement, 1=flying up, 2=flying down)
Source: arthur900530/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from hitframe_models.layers import CoordinateEmbedding
import numpy as np
import pickle


class OptimusPrime(nn.Module):
    """
    Transformer that predicts shuttlecock direction from player keypoints.
    Input: (batch, seq_len, 2, 17, 2) - 2 players, 17 keypoints, (x,y)
    Output: (seq_len, batch, 4) - 4 classes per frame: 0=pad, 1=up, 2=down, 3=no_move
    """
    def __init__(self, num_tokens=4, dim_model=2048, num_heads=8,
                 num_encoder_layers=8, dim_feedforward=2048, dropout_p=0):
        super().__init__()
        self.dim_model = dim_model
        
        # Coordinate embedding for player keypoints
        self.xy_embedding = CoordinateEmbedding(in_channels=34, emb_size=dim_model)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            dim_model, num_heads, dim_feedforward, dropout_p
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Decoder
        self.decoder1 = nn.Linear(dim_model, dim_model)
        self.decoder2 = nn.Linear(dim_model, num_tokens)

    def forward(self, src, src_pad_mask=None):
        src = self.xy_embedding(src)
        src = src.permute(1, 0, 2)  # (seq_len, batch, dim_model)
        
        output = self.transformer_encoder(src, src_key_padding_mask=src_pad_mask)
        output = F.relu(self.decoder1(output))
        output = self.decoder2(output)
        return output

    def create_src_pad_mask(self, matrix: torch.tensor,
                            PAD_array=np.zeros((1, 2, 17, 2))) -> torch.tensor:
        """Create padding mask for sequences shorter than max length."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        src_pad_mask = []
        PAD_array = torch.tensor(PAD_array).squeeze(0).to(device)
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                a = matrix[i][j].to(device)
                src_pad_mask.append(torch.equal(a, PAD_array))
        
        src_pad_mask = torch.tensor(src_pad_mask).unsqueeze(0).reshape(
            matrix.shape[0], -1
        ).to(device)
        return src_pad_mask


class OptimusPrimeContainer(object):
    """
    Container for the OPT transformer model.
    Handles loading, scaling, and prediction.
    """
    def __init__(self, opt_path, scaler_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__setup_model(opt_path)
        self.__setup_scaler(scaler_path)

    def __setup_model(self, opt_path):
        self.model = OptimusPrime(
            num_tokens=4, dim_model=2048, num_heads=8,
            num_encoder_layers=8, dim_feedforward=2048, dropout_p=0
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(opt_path, map_location=self.device)
        )
        self.model.eval()

    def __setup_scaler(self, scaler_path):
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def __scale(self, input_sequence):
        """Scale keypoints using the pre-fitted scaler."""
        input_sequence = np.array(input_sequence)
        temp = []
        for i in range(input_sequence.shape[0]):
            # Flatten (2, 17, 2) -> (1, 68) for scaler
            scaled_joint = self.scaler.transform(
                np.reshape(input_sequence[i], [1, -1])
            )
            # Reshape back to (2, 17, 2)
            temp.append(np.reshape(scaled_joint, [2, 17, 2]))
        
        input_sequence = torch.tensor(
            np.array(temp)
        ).unsqueeze(0).to(torch.float32).to(self.device)
        return input_sequence

    @torch.no_grad()
    def predict(self, input_sequence):
        """
        Predict shuttlecock direction for each frame in the sequence.
        Input: list of keypoint frames, each shape (2, 17, 2)
        Output: tensor of direction indices (0=pad, 1=up, 2=down, 3=no_move)
        """
        input_sequence = self.__scale(input_sequence)
        src_pad_mask = self.model.create_src_pad_mask(input_sequence)
        pred = self.model(input_sequence, src_pad_mask=src_pad_mask)
        pred_indices = torch.max(pred.detach(), 2).indices.squeeze(-1)
        return pred_indices
