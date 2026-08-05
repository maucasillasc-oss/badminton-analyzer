"""
TrackNetV3 Shuttlecock Tracker - Implementación correcta
Usa los pesos pre-entrenados exactamente como en el repositorio original.
"""
import torch
import numpy as np
import cv2
import os
import gdown
import zipfile
from tracknet_model import TrackNet, InpaintNet

# Configuración
TRACKNET_WEIGHTS_URL = "https://drive.google.com/uc?id=1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"
WEIGHTS_DIR = "ckpts"
TRACKNET_FILE = os.path.join(WEIGHTS_DIR, "TrackNet_best.pt")
INPAINTNET_FILE = os.path.join(WEIGHTS_DIR, "InpaintNet_best.pt")

# Dimensiones de entrada de TrackNet (fijas según el paper)
INPUT_H = 288
INPUT_W = 512


def download_weights():
    """Descarga los pesos pre-entrenados si no existen"""
    if os.path.exists(TRACKNET_FILE):
        return
    
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    zip_path = os.path.join(WEIGHTS_DIR, "TrackNetV3_ckpts.zip")
    
    if not os.path.exists(zip_path):
        print("Descargando pesos de TrackNetV3...")
        gdown.download(TRACKNET_WEIGHTS_URL, zip_path, quiet=False)
    
    # Descomprimir
    print("Descomprimiendo...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(WEIGHTS_DIR)
    
    # Buscar los archivos .pt en todas las subcarpetas
    for root, dirs, files in os.walk(WEIGHTS_DIR):
        for f in files:
            if f == 'TrackNet_best.pt' and root != WEIGHTS_DIR:
                os.rename(os.path.join(root, f), TRACKNET_FILE)
            elif f == 'InpaintNet_best.pt' and root != WEIGHTS_DIR:
                os.rename(os.path.join(root, f), INPAINTNET_FILE)
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    print(f"✓ Pesos en {WEIGHTS_DIR}")


class TrackNetTracker:
    """Tracker usando TrackNetV3 pre-entrenado"""
    
    def __init__(self):
        download_weights()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"TrackNet device: {self.device}")
        
        # Cargar checkpoint de TrackNet
        self.tracknet = None
        self.seq_len = 8
        self.bg_mode = 'concat'
        
        if os.path.exists(TRACKNET_FILE):
            ckpt = torch.load(TRACKNET_FILE, map_location=self.device)
            
            # Extraer parámetros del checkpoint
            if 'param_dict' in ckpt:
                self.seq_len = ckpt['param_dict'].get('seq_len', 8)
                self.bg_mode = ckpt['param_dict'].get('bg_mode', 'concat')
            
            # Calcular in_dim basado en bg_mode
            if self.bg_mode == 'concat':
                in_dim = (self.seq_len + 1) * 3  # frames + background concatenados
            elif self.bg_mode == 'subtract':
                in_dim = self.seq_len * 3
            else:
                in_dim = self.seq_len * 3 + 3
            
            # Crear modelo con dimensiones correctas
            self.tracknet = TrackNet(in_dim=in_dim, out_dim=self.seq_len)
            
            # Cargar pesos (el checkpoint guarda en 'model' key)
            if 'model' in ckpt:
                self.tracknet.load_state_dict(ckpt['model'])
            else:
                self.tracknet.load_state_dict(ckpt)
            
            self.tracknet.to(self.device)
            self.tracknet.eval()
            print(f"✓ TrackNet cargado (seq_len={self.seq_len}, bg_mode={self.bg_mode})")
        else:
            print("⚠ TrackNet_best.pt no encontrado")
        
        # Cargar InpaintNet
        self.inpaintnet = None
        if os.path.exists(INPAINTNET_FILE):
            ckpt = torch.load(INPAINTNET_FILE, map_location=self.device)
            self.inpaintnet = InpaintNet()
            if 'model' in ckpt:
                self.inpaintnet.load_state_dict(ckpt['model'])
            else:
                self.inpaintnet.load_state_dict(ckpt)
            self.inpaintnet.to(self.device)
            self.inpaintnet.eval()
            print("✓ InpaintNet cargado")
        
        self.positions = []
        self.background = None
    
    def estimate_background(self, video_path, num_samples=100):
        """Estima fondo usando mediana de frames (como en el paper)"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames), dtype=int)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                small = cv2.resize(frame, (INPUT_W, INPUT_H))
                # BGR to RGB como en el original
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
        
        cap.release()
        
        if frames:
            self.background = np.median(frames, axis=0).astype(np.uint8)
        else:
            self.background = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
    
    def process_video(self, video_path, progress_callback=None):
        """Procesa video completo y devuelve posiciones del volante"""
        self.estimate_background(video_path)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.positions = []
        frame_buffer = []
        frame_count = 0
        
        # Background como tensor
        bg_tensor = torch.FloatTensor(self.background).permute(2, 0, 1) / 255.0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Redimensionar y convertir a RGB
            small = cv2.resize(frame, (INPUT_W, INPUT_H))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            frame_buffer.append(rgb)
            
            # Procesar cuando tenemos seq_len frames
            if len(frame_buffer) == self.seq_len:
                if self.tracknet is not None:
                    batch_positions = self._predict_batch(frame_buffer, bg_tensor)
                    self.positions.extend(batch_positions)
                else:
                    # Sin modelo, agregar None
                    self.positions.extend([None] * self.seq_len)
                
                frame_buffer = []
            
            frame_count += 1
            if progress_callback and frame_count % 200 == 0:
                progress_callback(int((frame_count / total_frames) * 50))
        
        # Procesar frames restantes
        if frame_buffer and self.tracknet is not None:
            while len(frame_buffer) < self.seq_len:
                frame_buffer.append(frame_buffer[-1])
            batch_positions = self._predict_batch(frame_buffer, bg_tensor)
            self.positions.extend(batch_positions[:len(frame_buffer)])
        
        cap.release()
        
        return {
            'positions': self.positions,
            'fps': fps,
            'total_frames': total_frames
        }
    
    @torch.no_grad()
    def _predict_batch(self, frames, bg_tensor):
        """Predice posiciones del volante para seq_len frames"""
        # Construir input según bg_mode
        frame_tensors = []
        for frame in frames:
            t = torch.FloatTensor(frame).permute(2, 0, 1) / 255.0
            frame_tensors.append(t)
        
        if self.bg_mode == 'concat':
            # Background va PRIMERO, luego los frames (como en el dataset original)
            input_tensor = torch.cat([bg_tensor] + frame_tensors, dim=0)
        elif self.bg_mode == 'subtract':
            # Restar background de cada frame
            subtracted = [f - bg_tensor for f in frame_tensors]
            input_tensor = torch.cat(subtracted, dim=0)
        else:
            # Default: frames + background separado
            input_tensor = torch.cat(frame_tensors + [bg_tensor], dim=0)
        
        input_tensor = input_tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)
        
        # Predicción
        output = self.tracknet(input_tensor)  # (1, seq_len, H, W)
        
        # Extraer posiciones
        positions = []
        for i in range(self.seq_len):
            heatmap = output[0, i].cpu().numpy()
            pos = self._heatmap_to_position(heatmap)
            positions.append(pos)
        
        return positions
    
    def _heatmap_to_position(self, heatmap, threshold=0.5):
        """Convierte heatmap a posición (x, y) del volante"""
        if heatmap.max() < threshold:
            return None  # No detectado
        
        # Binarizar
        binary = (heatmap > threshold).astype(np.uint8) * 255
        
        # Encontrar contornos para centro más preciso
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Usar el contorno más grande
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy, float(heatmap.max()))
        
        # Fallback: posición del máximo
        y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        return (int(x), int(y), float(heatmap.max()))
    
    def detect_shots(self, min_frames_between=20):
        """Detecta golpes por cambio de dirección del volante"""
        shots = []
        last_shot_idx = -min_frames_between
        
        # Obtener posiciones válidas con índice
        valid = [(i, p[0], p[1]) for i, p in enumerate(self.positions) if p is not None]
        
        if len(valid) < 6:
            return shots
        
        for i in range(4, len(valid) - 1):
            frame_idx = valid[i][0]
            
            if (frame_idx - last_shot_idx) < min_frames_between:
                continue
            
            # Vectores de velocidad (usando ventana de 2 frames)
            # Velocidad actual
            dx1 = valid[i][1] - valid[i-1][1]
            dy1 = valid[i][2] - valid[i-1][2]
            
            # Velocidad anterior (2 frames antes)
            dx2 = valid[i-2][1] - valid[i-3][1]
            dy2 = valid[i-2][2] - valid[i-3][2]
            
            speed = np.sqrt(dx1**2 + dy1**2)
            
            # Cambio de dirección vertical
            vertical_change = abs(dy1 - dy2)
            
            # Cambio de signo vertical (subir↔bajar)
            sign_change_y = (dy1 * dy2) < 0 and abs(dy1) > 3 and abs(dy2) > 3
            
            # Golpe = cambio significativo de dirección + velocidad mínima
            if (sign_change_y or vertical_change > 20) and speed > 5:
                shots.append({
                    'frame_idx': frame_idx,
                    'x': valid[i][1],
                    'y': valid[i][2],
                    'speed': speed,
                    'direction': 'down' if dy1 > 0 else 'up'
                })
                last_shot_idx = frame_idx
        
        return shots
