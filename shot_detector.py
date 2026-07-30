import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

class ShotDetector:
    """Detecta golpes usando YOLO para jugadores + tracking del volante"""
    
    def __init__(self):
        # YOLO para detectar personas
        self.model = YOLO('yolov8n.pt')  # Se descarga automáticamente
        
        # Parámetros de tracking del volante
        self.shuttlecock_history = deque(maxlen=10)
        self.prev_frame_gray = None
        self.shots = []
        self.player_positions = {'top': [], 'bottom': []}
        
        # Parámetros para detección de golpes
        self.min_frames_between_shots = 8  # ~0.27 seg a 30fps
        self.last_shot_frame = -self.min_frames_between_shots
    
    def process_video(self, video_path, progress_callback=None):
        """Procesa el video completo y devuelve los golpes detectados"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # Zona de la cancha central (ignorar laterales)
        court_left = int(frame_w * 0.15)
        court_right = int(frame_w * 0.85)
        mid_y = frame_h // 2  # Línea divisoria entre J1 (abajo) y J2 (arriba)
        
        frame_count = 0
        rallies = []
        current_rally_shots = 0
        frames_without_action = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar cada 3 frames para velocidad con GPU
            if frame_count % 3 == 0:
                # Recortar a cancha central
                court_frame = frame[:, court_left:court_right]
                
                # Detectar jugadores con YOLO
                players = self._detect_players(court_frame, mid_y)
                
                # Detectar volante por movimiento rápido
                shuttlecock_pos = self._detect_shuttlecock(court_frame)
                
                # Detectar si hubo golpe
                shot = self._detect_shot(shuttlecock_pos, players, frame_count, fps, mid_y, frame_h)
                
                if shot:
                    self.shots.append(shot)
                    current_rally_shots += 1
                    frames_without_action = 0
                else:
                    frames_without_action += 1
                
                # Detectar fin de rally (3+ segundos sin golpe)
                if current_rally_shots > 0 and frames_without_action > fps * 3:
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
        results = self.model(frame, verbose=False, conf=0.4, classes=[0])  # clase 0 = persona
        
        players = {'top': None, 'bottom': None}
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                foot_y = y2  # Pies del jugador
                
                if foot_y > mid_y:
                    players['bottom'] = {'x': center_x, 'y': foot_y, 'bbox': (x1, y1, x2, y2)}
                else:
                    players['top'] = {'x': center_x, 'y': foot_y, 'bbox': (x1, y1, x2, y2)}
        
        return players
    
    def _detect_shuttlecock(self, frame):
        """Detecta el volante usando diferencia de frames + detección de objetos pequeños"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return None
        
        # Diferencia entre frames
        diff = cv2.absdiff(gray, self.prev_frame_gray)
        _, thresh = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)
        
        # Morfología para limpiar
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Encontrar objetos pequeños y rápidos (volante)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        min_area = 15
        max_area = 400
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                # El volante es aproximadamente circular
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 3.0:
                    center = (x + w // 2, y + h // 2)
                    if best_candidate is None or area < best_candidate['area']:
                        best_candidate = {'pos': center, 'area': area}
        
        self.prev_frame_gray = gray
        
        if best_candidate:
            self.shuttlecock_history.append(best_candidate['pos'])
            return best_candidate['pos']
        
        return None
    
    def _detect_shot(self, shuttlecock_pos, players, frame_count, fps, mid_y, frame_h):
        """Detecta si ocurrió un golpe basado en cambio de dirección del volante"""
        
        if len(self.shuttlecock_history) < 4:
            return None
        
        if (frame_count - self.last_shot_frame) < self.min_frames_between_shots:
            return None
        
        # Calcular dirección del volante en los últimos frames
        recent = list(self.shuttlecock_history)
        
        # Vector de movimiento reciente
        dx = recent[-1][0] - recent[-3][0]
        dy = recent[-1][1] - recent[-3][1]
        
        # Vector de movimiento anterior
        prev_dx = recent[-3][0] - recent[-4][0] if len(recent) > 3 else 0
        prev_dy = recent[-3][1] - recent[-4][1] if len(recent) > 3 else 0
        
        # Detectar cambio brusco de dirección (golpe)
        speed = np.sqrt(dx**2 + dy**2)
        direction_change = abs(dy - prev_dy) + abs(dx - prev_dx)
        
        if speed > 10 and direction_change > 15:
            self.last_shot_frame = frame_count
            
            # Determinar quién golpeó
            shuttle_y = recent[-1][1]
            player = 'J1' if shuttle_y > mid_y * 0.6 else 'J2'
            
            # Clasificar tipo de golpe por trayectoria
            shot_type = self._classify_shot(dx, dy, speed, shuttle_y, mid_y, frame_h)
            
            # Determinar zona
            shuttle_x = recent[-1][0]
            zone = self._get_zone(shuttle_x, shuttle_y, mid_y, frame_h)
            
            return {
                'frame': frame_count,
                'timestamp': round(frame_count / fps, 1),
                'player': player,
                'type': shot_type,
                'speed': speed,
                'zone': zone
            }
        
        return None
    
    def _classify_shot(self, dx, dy, speed, shuttle_y, mid_y, frame_h):
        """Clasifica el tipo de golpe por trayectoria"""
        
        # Smash: muy rápido y hacia abajo
        if speed > 40 and dy > 20:
            return 'smash'
        
        # Clear: hacia arriba con velocidad
        if dy < -15 and speed > 20:
            return 'clear'
        
        # Net: cerca de la red (centro del frame) y lento
        net_zone = abs(shuttle_y - mid_y) < (frame_h * 0.15)
        if net_zone and speed < 25:
            return 'net'
        
        # Drop: velocidad media, bajando suavemente
        if 10 < speed < 30 and dy > 5:
            return 'drop'
        
        # Drive: horizontal rápido
        if abs(dx) > abs(dy) * 1.3 and speed > 20:
            return 'drive'
        
        return 'other'
    
    def _get_zone(self, x, y, mid_y, frame_h):
        """Determina la zona de la cancha"""
        # Dividir en 6 zonas
        third_x = x / 3  # Normalizar por ancho
        
        if y < mid_y:
            prefix = 'front'
        else:
            prefix = 'back'
        
        if third_x < 0.33:
            suffix = '_left'
        elif third_x < 0.66:
            suffix = '_center'
        else:
            suffix = '_right'
        
        return prefix + suffix
