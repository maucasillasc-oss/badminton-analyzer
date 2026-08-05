import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from tracknet import ShuttlecockTracker

class ShotDetector:
    """Detecta golpes usando YOLO para jugadores + TrackNet para volante"""
    
    def __init__(self):
        # YOLO para detectar personas
        self.model = YOLO('yolov8n.pt')
        
        # TrackNet para tracking del volante
        self.tracker = ShuttlecockTracker()
        
        # Parámetros de detección de golpes
        self.shots = []
        self.min_frames_between_shots = 15  # ~0.5 seg a 30fps
        self.last_shot_frame = -self.min_frames_between_shots
    
    def process_video(self, video_path, progress_callback=None):
        """Procesa el video completo y devuelve los golpes detectados"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # Zona de la cancha central
        court_left = int(frame_w * 0.15)
        court_right = int(frame_w * 0.85)
        mid_y = frame_h // 2
        
        frame_count = 0
        rallies = []
        current_rally_shots = 0
        frames_without_shot = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Recortar a cancha central
            court_frame = frame[:, court_left:court_right]
            
            # TrackNet: detectar volante en cada frame
            shuttle_pos = self.tracker.process_frame(court_frame)
            
            # Cada 3 frames: detectar jugadores con YOLO
            players = None
            if frame_count % 3 == 0:
                players = self._detect_players(court_frame, mid_y)
            
            # Detectar golpe por cambio de dirección del volante
            if (frame_count - self.last_shot_frame) >= self.min_frames_between_shots:
                is_shot = self.tracker.detect_shot(min_direction_change=25)
                
                if is_shot and shuttle_pos:
                    # Determinar quién golpeó
                    player = 'J1' if shuttle_pos[1] > mid_y * 0.5 else 'J2'
                    
                    # Clasificar tipo de golpe
                    direction = self.tracker.get_trajectory_direction()
                    speed = self._get_shuttle_speed()
                    shot_type = self._classify_shot(direction, speed, shuttle_pos, mid_y, frame_h)
                    
                    # Zona
                    zone = self._get_zone(shuttle_pos[0], shuttle_pos[1], mid_y, court_right - court_left)
                    
                    self.shots.append({
                        'frame': frame_count,
                        'timestamp': round(frame_count / fps, 1),
                        'player': player,
                        'type': shot_type,
                        'speed': speed,
                        'zone': zone
                    })
                    
                    self.last_shot_frame = frame_count
                    current_rally_shots += 1
                    frames_without_shot = 0
                else:
                    frames_without_shot += 1
            
            # Detectar fin de rally (2.5+ segundos sin golpe)
            if current_rally_shots > 0 and frames_without_shot > fps * 2.5:
                rallies.append({'length': current_rally_shots})
                current_rally_shots = 0
            
            frame_count += 1
            
            if progress_callback and frame_count % 100 == 0:
                progress_callback(int((frame_count / total_frames) * 70))
        
        # Cerrar último rally
        if current_rally_shots > 0:
            rallies.append({'length': current_rally_shots})
        
        cap.release()
        
        return {
            'shots': self.shots,
            'rallies': rallies,
            'fps': fps,
            'total_frames': total_frames
        }
    
    def _detect_players(self, frame, mid_y):
        """Detecta jugadores con YOLO"""
        results = self.model(frame, verbose=False, conf=0.4, classes=[0])
        players = {'top': None, 'bottom': None}
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                foot_y = y2
                center_x = (x1 + x2) / 2
                
                if foot_y > mid_y:
                    players['bottom'] = {'x': center_x, 'y': foot_y}
                else:
                    players['top'] = {'x': center_x, 'y': foot_y}
        
        return players
    
    def _get_shuttle_speed(self):
        """Calcula velocidad del volante"""
        positions = self.tracker.positions
        if len(positions) < 3:
            return 0
        dx = positions[-1][0] - positions[-3][0]
        dy = positions[-1][1] - positions[-3][1]
        return np.sqrt(dx**2 + dy**2)
    
    def _classify_shot(self, direction, speed, shuttle_pos, mid_y, frame_h):
        """Clasifica el tipo de golpe"""
        net_zone = abs(shuttle_pos[1] - mid_y) < (frame_h * 0.15)
        
        # Smash: muy rápido y hacia abajo
        if speed > 35 and direction == 'down':
            return 'smash'
        
        # Clear: hacia arriba con velocidad
        if direction == 'up' and speed > 15:
            return 'clear'
        
        # Net: cerca de la red y lento
        if net_zone and speed < 20:
            return 'net'
        
        # Drop: bajando suavemente
        if direction == 'down' and speed < 25:
            return 'drop'
        
        # Drive: horizontal
        if direction in ('left', 'right') and speed > 15:
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
