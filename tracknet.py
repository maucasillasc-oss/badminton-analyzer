"""
TrackNetV3 Shuttlecock Tracker
Usa el modelo TrackNetV3 pre-entrenado para detectar el volante frame a frame.
Los pesos se descargan de Google Drive al iniciar.
"""
import torch
import numpy as np
import cv2
import os
import gdown
from tracknet_model import TrackNet, InpaintNet

# Configuración
TRACKNET_WEIGHTS_URL = "https://drive.google.com/uc?id=1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"
WEIGHTS_DIR = "ckpts"
TRACKNET_FILE = os.path.join(WEIGHTS_DIR, "TrackNet_best.pt")
INPAINTNET_FILE = os.path.join(WEIGHTS_DIR, "InpaintNet_best.pt")

# Dimensiones de entrada de TrackNet
INPUT_H = 288
INPUT_W = 512
SEQ_LEN = 8  # TrackNetV3 usa secuencias de 8 frames


def download_weights():
    """Descarga los pesos pre-entrenados si no existen"""
    if os.path.exists(TRACKNET_FILE) and os.path.exists(INPAINTNET_FILE):
        return
    
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    zip_path = os.path.join(WEIGHTS_DIR, "TrackNetV3_ckpts.zip")
    
    print("Descargando pesos de TrackNetV3...")
    gdown.download(TRACKNET_WEIGHTS_URL, zip_path, quiet=False)
    
    # Descomprimir
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(WEIGHTS_DIR)
    
    # Mover archivos si están en subcarpeta
    for root, dirs, files in os.walk(WEIGHTS_DIR):
        for f in files:
            if f.endswith('.pt'):
                src = os.path.join(root, f)
                dst = os.path.join(WEIGHTS_DIR, f)
                if src != dst:
                    os.rename(src, dst)
    
    # Limpiar zip
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    print("✓ Pesos descargados")


class TrackNetTracker:
    """Tracker de volante usando TrackNetV3 pre-entrenado"""
    
    def __init__(self):
        download_weights()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"TrackNet usando: {self.device}")
        
        # Cargar modelo TrackNet
        # in_dim = SEQ_LEN * 3 (RGB) + 3 (background) = 27
        # out_dim = SEQ_LEN = 8
        self.tracknet = TrackNet(in_dim=SEQ_LEN * 3 + 3, out_dim=SEQ_LEN)
        
        if os.path.exists(TRACKNET_FILE):
            checkpoint = torch.load(TRACKNET_FILE, map_location=self.device)
            self.tracknet.load_state_dict(checkpoint)
            print("✓ TrackNet cargado")
        else:
            print("⚠ Pesos de TrackNet no encontrados, usando sin pre-entrenar")
        
        self.tracknet.to(self.device)
        self.tracknet.eval()
        
        # Cargar InpaintNet para rectificación de trayectoria
        self.inpaintnet = InpaintNet()
        if os.path.exists(INPAINTNET_FILE):
            checkpoint = torch.load(INPAINTNET_FILE, map_location=self.device)
            self.inpaintnet.load_state_dict(checkpoint)
            print("✓ InpaintNet cargado")
        
        self.inpaintnet.to(self.device)
        self.inpaintnet.eval()
        
        # Buffer de frames
        self.frame_buffer = []
        self.background = None
        self.positions = []  # Lista de (x, y, visibility) por frame
    
    def estimate_background(self, video_path, num_samples=50):
        """Estima el fondo del video usando la mediana de frames"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Seleccionar frames uniformemente distribuidos
        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                small = cv2.resize(frame, (INPUT_W, INPUT_H))
                frames.append(small)
        
        cap.release()
        
        if frames:
            # Mediana para estimar background (elimina objetos en movimiento)
            self.background = np.median(frames, axis=0).astype(np.uint8)
        else:
            self.background = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
    
    def process_video(self, video_path, progress_callback=None):
        """Procesa el video completo y devuelve posiciones del volante"""
        # Estimar background
        self.estimate_background(video_path)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.positions = []
        self.frame_buffer = []
        frame_count = 0
        
        # Background normalizado
        bg_tensor = self._frame_to_tensor(self.background)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Redimensionar
            small = cv2.resize(frame, (INPUT_W, INPUT_H))
            self.frame_buffer.append(small)
            
            # Cuando tenemos suficientes frames, procesar batch
            if len(self.frame_buffer) == SEQ_LEN:
                positions_batch = self._predict_batch(bg_tensor)
                self.positions.extend(positions_batch)
                
                # Mantener último frame para solapamiento
                self.frame_buffer = self.frame_buffer[SEQ_LEN:]
            
            frame_count += 1
            if progress_callback and frame_count % 100 == 0:
                progress_callback(int((frame_count / total_frames) * 50))
        
        # Procesar frames restantes
        if self.frame_buffer:
            # Pad con el último frame hasta completar SEQ_LEN
            while len(self.frame_buffer) < SEQ_LEN:
                self.frame_buffer.append(self.frame_buffer[-1])
            positions_batch = self._predict_batch(bg_tensor)
            self.positions.extend(positions_batch[:len(self.frame_buffer)])
        
        cap.release()
        
        return {
            'positions': self.positions,
            'fps': fps,
            'total_frames': total_frames
        }
    
    def _frame_to_tensor(self, frame):
        """Convierte frame BGR a tensor normalizado"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.FloatTensor(rgb).permute(2, 0, 1) / 255.0
        return tensor
    
    @torch.no_grad()
    def _predict_batch(self, bg_tensor):
        """Predice posiciones del volante para un batch de SEQ_LEN frames"""
        # Construir input: concatenar frames + background
        frame_tensors = []
        for frame in self.frame_buffer[:SEQ_LEN]:
            frame_tensors.append(self._frame_to_tensor(frame))
        
        # Input shape: (1, SEQ_LEN*3 + 3, H, W)
        input_tensor = torch.cat(frame_tensors + [bg_tensor], dim=0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Predicción
        output = self.tracknet(input_tensor)  # (1, SEQ_LEN, H, W)
        
        # Extraer posiciones de cada frame
        positions = []
        for i in range(SEQ_LEN):
            heatmap = output[0, i].cpu().numpy()
            pos = self._heatmap_to_position(heatmap)
            positions.append(pos)
        
        return positions
    
    def _heatmap_to_position(self, heatmap, threshold=0.5):
        """Convierte heatmap a coordenada (x, y)"""
        if heatmap.max() < threshold:
            return None  # No se detectó volante
        
        # Encontrar el máximo
        y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        
        # Escalar a resolución original no es necesario aquí,
        # se hace después según la resolución del video
        return (int(x), int(y), float(heatmap.max()))
    
    def detect_shots(self, min_frames_between=20):
        """Detecta golpes basado en cambios de dirección del volante"""
        shots = []
        last_shot_idx = -min_frames_between
        
        # Filtrar posiciones válidas
        valid_positions = []
        for i, pos in enumerate(self.positions):
            if pos is not None:
                valid_positions.append((i, pos[0], pos[1]))
        
        if len(valid_positions) < 5:
            return shots
        
        # Detectar cambios de dirección
        for i in range(4, len(valid_positions)):
            frame_idx = valid_positions[i][0]
            
            if (frame_idx - last_shot_idx) < min_frames_between:
                continue
            
            # Vectores de movimiento
            # Actual
            dx1 = valid_positions[i][1] - valid_positions[i-1][1]
            dy1 = valid_positions[i][2] - valid_positions[i-1][2]
            
            # Anterior
            dx2 = valid_positions[i-2][1] - valid_positions[i-3][1]
            dy2 = valid_positions[i-2][2] - valid_positions[i-3][2]
            
            # Velocidad
            speed = np.sqrt(dx1**2 + dy1**2)
            
            # Cambio de dirección vertical (clave en badminton)
            vertical_change = abs(dy1 - dy2)
            
            # Cambio de signo en Y (volante cambió de subir a bajar o viceversa)
            sign_change = (dy1 * dy2) < 0
            
            # Golpe = cambio significativo de dirección vertical + velocidad
            if (vertical_change > 15 or sign_change) and speed > 8:
                shots.append({
                    'frame_idx': frame_idx,
                    'x': valid_positions[i][1],
                    'y': valid_positions[i][2],
                    'speed': speed,
                    'direction': 'down' if dy1 > 0 else 'up'
                })
                last_shot_idx = frame_idx
        
        return shots
