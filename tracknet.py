import torch
import torch.nn as nn
import cv2
import numpy as np

class TrackNetModel(nn.Module):
    """TrackNet: modelo para tracking de volante en badminton
    Basado en el paper TrackNet (2019) - simplificado para inferencia"""
    
    def __init__(self):
        super(TrackNetModel, self).__init__()
        
        # Encoder (3 frames RGB como input = 9 canales)
        self.encoder = nn.Sequential(
            nn.Conv2d(9, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ShuttlecockTracker:
    """Tracker de volante usando detección por movimiento optimizada"""
    
    def __init__(self):
        self.input_h = 288
        self.input_w = 512
        self.positions = []
        self.prev_frames = []
    
    def process_frame(self, frame):
        """Procesa un frame y devuelve la posición del volante"""
        small = cv2.resize(frame, (self.input_w, self.input_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        self.prev_frames.append(gray)
        
        # Necesitamos al menos 3 frames
        if len(self.prev_frames) < 3:
            return None
        
        # Mantener solo los últimos 3
        if len(self.prev_frames) > 3:
            self.prev_frames.pop(0)
        
        # Detectar el volante por diferencia temporal multi-frame
        pos = self._detect_shuttlecock_multiframe()
        
        if pos:
            # Escalar de vuelta a resolución original
            scale_x = frame.shape[1] / self.input_w
            scale_y = frame.shape[0] / self.input_h
            real_pos = (int(pos[0] * scale_x), int(pos[1] * scale_y))
            self.positions.append(real_pos)
            return real_pos
        
        return None
    
    def _detect_shuttlecock_multiframe(self):
        """Detecta volante usando diferencia entre 3 frames consecutivos"""
        f1, f2, f3 = self.prev_frames[-3], self.prev_frames[-2], self.prev_frames[-1]
        
        # Diferencia temporal: objetos que se mueven rápido aparecen en ambas diferencias
        diff1 = cv2.absdiff(f2, f1)
        diff2 = cv2.absdiff(f3, f2)
        
        # AND: solo objetos que se movieron en AMBOS intervalos (movimiento continuo)
        combined = cv2.bitwise_and(diff1, diff2)
        
        # Umbral alto para solo objetos muy rápidos
        _, thresh = cv2.threshold(combined, 40, 255, cv2.THRESH_BINARY)
        
        # Erosión + dilatación para limpiar ruido
        kernel_small = np.ones((2, 2), np.uint8)
        kernel_big = np.ones((4, 4), np.uint8)
        thresh = cv2.erode(thresh, kernel_small, iterations=1)
        thresh = cv2.dilate(thresh, kernel_big, iterations=1)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar: volante es pequeño (5-200 px²) y relativamente circular
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 200:
                x, y, w, h = cv2.boundingRect(contour)
                aspect = w / h if h > 0 else 0
                if 0.2 < aspect < 5.0:  # No muy alargado
                    center = (x + w // 2, y + h // 2)
                    candidates.append({'pos': center, 'area': area})
        
        if not candidates:
            return None
        
        # Si hay posiciones previas, elegir el candidato más cercano al último
        if self.positions:
            last = self.positions[-1]
            # Escalar last a resolución pequeña
            scale_x = self.input_w / (last[0] / (self.positions[-1][0] / last[0]) if last[0] > 0 else self.input_w)
            # Simplificar: elegir el más pequeño (volante es pequeño)
            best = min(candidates, key=lambda c: c['area'])
            return best['pos']
        else:
            # Primera detección: elegir el más pequeño
            best = min(candidates, key=lambda c: c['area'])
            return best['pos']
    
    def detect_shot(self, min_direction_change=20):
        """Detecta si hubo un golpe basado en cambio de dirección del volante"""
        if len(self.positions) < 5:
            return False
        
        recent = self.positions[-5:]
        
        # Vector de movimiento actual
        dx1 = recent[-1][0] - recent[-2][0]
        dy1 = recent[-1][1] - recent[-2][1]
        
        # Vector de movimiento anterior
        dx2 = recent[-3][0] - recent[-4][0]
        dy2 = recent[-3][1] - recent[-4][1]
        
        # Cambio de dirección vertical (más importante en badminton)
        vertical_change = abs(dy1 - dy2)
        
        # Velocidad actual
        speed = np.sqrt(dx1**2 + dy1**2)
        
        # Un golpe = cambio de dirección vertical significativo + velocidad
        if vertical_change > min_direction_change and speed > 5:
            return True
        
        return False
    
    def get_trajectory_direction(self):
        """Devuelve la dirección de la trayectoria actual"""
        if len(self.positions) < 3:
            return 'unknown'
        
        dy = self.positions[-1][1] - self.positions[-3][1]
        dx = self.positions[-1][0] - self.positions[-3][0]
        speed = np.sqrt(dx**2 + dy**2)
        
        if speed < 3:
            return 'still'
        
        if abs(dy) > abs(dx):
            return 'down' if dy > 0 else 'up'
        else:
            return 'right' if dx > 0 else 'left'
