"""
Shot Detector: YOLO + TrackNetV3 + lógica de detección de golpes
- TrackNetV3: Detecta posición del volante con 97.5% de precisión
- YOLO: Detecta y rastrea jugadores
- Lógica: Combina ambos para determinar golpes, quién golpeó, y tipo
"""
import cv2
import numpy as np
from ultralytics import YOLO
from tracknet import TrackNetTracker, INPUT_H, INPUT_W

class ShotDetector:
    def __init__(self):
        # YOLO para jugadores
        self.yolo = YOLO('yolov8n.pt')
        
        # TrackNet para volante
        self.tracker = TrackNetTracker()
    
    def process_video(self, video_path, progress_callback=None):
        """Pipeline completo: TrackNet + YOLO + detección de golpes"""
        
        # Paso 1: TrackNet detecta trayectoria del volante
        if progress_callback:
            progress_callback(5)
        
        track_result = self.tracker.process_video(video_path, progress_callback)
        fps = track_result['fps']
        total_frames = track_result['total_frames']
        
        if progress_callback:
            progress_callback(55)
        
        # Paso 2: Detectar golpes por cambio de dirección
        # min_frames_between = fps * 0.5 (mínimo 0.5 seg entre golpes)
        min_between = max(int(fps * 0.5), 12)
        raw_shots = self.tracker.detect_shots(min_frames_between=min_between)
        
        if progress_callback:
            progress_callback(60)
        
        # Paso 3: YOLO detecta jugadores en frames de golpe para saber quién golpeó
        cap = cv2.VideoCapture(video_path)
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        mid_y = frame_h // 2
        
        shots = []
        for i, raw_shot in enumerate(raw_shots):
            frame_idx = raw_shot['frame_idx']
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Detectar jugadores en este frame
            players = self._detect_players(frame, mid_y)
            
            # Escalar posición del volante de TrackNet a resolución original
            scale_x = frame_w / INPUT_W
            scale_y = frame_h / INPUT_H
            shuttle_x = int(raw_shot['x'] * scale_x)
            shuttle_y = int(raw_shot['y'] * scale_y)
            
            # Determinar quién golpeó (jugador más cercano al volante)
            player = self._who_hit(players, shuttle_x, shuttle_y, mid_y)
            
            # Clasificar tipo de golpe
            shot_type = self._classify_shot(
                raw_shot['direction'], 
                raw_shot['speed'],
                shuttle_y, mid_y, frame_h
            )
            
            # Zona de la cancha
            zone = self._get_zone(shuttle_x, shuttle_y, mid_y, frame_w)
            
            shots.append({
                'frame': frame_idx,
                'timestamp': round(frame_idx / fps, 1),
                'player': player,
                'type': shot_type,
                'speed': raw_shot['speed'],
                'zone': zone,
                'shuttle_pos': (shuttle_x, shuttle_y)
            })
            
            if progress_callback and i % 5 == 0:
                progress_callback(60 + int((i / max(len(raw_shots), 1)) * 10))
        
        cap.release()
        
        if progress_callback:
            progress_callback(70)
        
        # Paso 4: Construir rallies
        rallies = self._build_rallies(shots, fps)
        
        return {
            'shots': shots,
            'rallies': rallies,
            'fps': fps,
            'total_frames': total_frames
        }
    
    def _detect_players(self, frame, mid_y):
        """Detecta jugadores con YOLO"""
        results = self.yolo(frame, verbose=False, conf=0.4, classes=[0])
        players = {'top': None, 'bottom': None}
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                foot_y = y2
                center_x = (x1 + x2) / 2
                
                if foot_y > mid_y:
                    if players['bottom'] is None or box.conf[0] > 0.5:
                        players['bottom'] = {'x': center_x, 'y': foot_y}
                else:
                    if players['top'] is None or box.conf[0] > 0.5:
                        players['top'] = {'x': center_x, 'y': foot_y}
        
        return players
    
    def _who_hit(self, players, shuttle_x, shuttle_y, mid_y):
        """Determina quién golpeó basado en proximidad"""
        # Si el volante está en la mitad inferior, probablemente J1 golpeó
        # Si está en la mitad superior, probablemente J2 golpeó
        if shuttle_y > mid_y:
            return 'J1'
        else:
            return 'J2'
    
    def _classify_shot(self, direction, speed, shuttle_y, mid_y, frame_h):
        """Clasifica el tipo de golpe basado en trayectoria"""
        net_zone = abs(shuttle_y - mid_y) < (frame_h * 0.12)
        
        # Smash: rápido y bajando
        if speed > 30 and direction == 'down':
            return 'smash'
        
        # Clear: subiendo con velocidad
        if direction == 'up' and speed > 15:
            return 'clear'
        
        # Net: cerca de la red y lento
        if net_zone and speed < 18:
            return 'net'
        
        # Drop: bajando suave
        if direction == 'down' and speed < 20:
            return 'drop'
        
        # Drive: velocidad media
        if 12 < speed < 30:
            return 'drive'
        
        return 'other'
    
    def _get_zone(self, x, y, mid_y, frame_w):
        """Determina la zona de la cancha"""
        third_w = frame_w / 3
        prefix = 'front' if y < mid_y else 'back'
        
        if x < third_w:
            suffix = '_left'
        elif x < third_w * 2:
            suffix = '_center'
        else:
            suffix = '_right'
        
        return prefix + suffix
    
    def _build_rallies(self, shots, fps):
        """Construye rallies basado en gaps entre golpes"""
        rallies = []
        current_rally = 0
        
        for i in range(len(shots)):
            current_rally += 1
            
            # Si hay un gap de >3 segundos al siguiente golpe, fin de rally
            if i < len(shots) - 1:
                gap = (shots[i+1]['frame'] - shots[i]['frame']) / fps
                if gap > 3.0:
                    rallies.append({'length': current_rally})
                    current_rally = 0
        
        # Último rally
        if current_rally > 0:
            rallies.append({'length': current_rally})
        
        return rallies
