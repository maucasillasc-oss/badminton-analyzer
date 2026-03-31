import cv2
import numpy as np

class BadmintonDetectorImproved:
    def __init__(self):
        self.prev_frame = None
        self.court_bounds = None
        # Usar detector de personas de OpenCV (más ligero que YOLO)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
    def detect(self, frame):
        """Detecta jugadores, volante y elementos del juego"""
        detections = {
            'players': [],
            'shuttlecock': None,
            'score': None,
            'motion_intensity': 0
        }
        
        # Detectar jugadores con HOG (más preciso que diferencia de frames)
        detections['players'] = self._detect_players_hog(frame)
        
        # Detectar volante
        detections['shuttlecock'] = self._detect_shuttlecock(frame)
        
        # Calcular intensidad de movimiento
        detections['motion_intensity'] = self._calculate_motion(frame)
        
        self.prev_frame = frame.copy()
        return detections
    
    def _detect_players_hog(self, frame):
        """Detecta jugadores usando HOG (Histogram of Oriented Gradients)"""
        players = []
        h, w = frame.shape[:2]
        
        # Definir región de la cancha
        court_top = int(h * 0.2)
        court_bottom = int(h * 0.9)
        court_left = int(w * 0.1)
        court_right = int(w * 0.9)
        
        # Redimensionar para velocidad
        scale = 1.0  # Ya viene redimensionado
        small_frame = frame
        
        # Detectar personas
        try:
            boxes, weights = self.hog.detectMultiScale(
                small_frame,
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05,
                hitThreshold=0.5
            )
            
            for (x, y, w_box, h_box), weight in zip(boxes, weights):
                # Ya no necesita escalar
                center_x = x + w_box // 2
                center_y = y + h_box  # Pies del jugador
                
                # Filtrar solo jugadores en la cancha
                if (court_left < center_x < court_right and 
                    court_top < center_y < court_bottom):
                    players.append({
                        'x': center_x,
                        'y': center_y,
                        'area': w_box * h_box,
                        'confidence': float(weight)
                    })
        except:
            # Si HOG falla, usar método de respaldo
            players = self._detect_players_fallback(frame)
        
        return players
    
    def _detect_players_fallback(self, frame):
        """Método de respaldo para detectar jugadores"""
        players = []
        h, w = frame.shape[:2]
        
        court_top = int(h * 0.2)
        court_bottom = int(h * 0.9)
        court_left = int(w * 0.1)
        court_right = int(w * 0.9)
        
        if self.prev_frame is not None:
            diff = cv2.absdiff(frame, self.prev_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            
            mask = np.zeros_like(thresh)
            mask[court_top:court_bottom, court_left:court_right] = 255
            thresh = cv2.bitwise_and(thresh, mask)
            
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 2000 < area < 40000:
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    
                    if h_box > w_box * 0.8:
                        center_x = x + w_box // 2
                        center_y = y + h_box
                        
                        if court_left < center_x < court_right and court_top < center_y < court_bottom:
                            players.append({'x': center_x, 'y': center_y, 'area': area})
        
        return players
    
    def _detect_shuttlecock(self, frame):
        """Detecta el volante usando detección de objetos pequeños y rápidos"""
        if self.prev_frame is None:
            return None
        
        diff = cv2.absdiff(frame, self.prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray_diff, 80, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 300:
                x, y, w, h = cv2.boundingRect(contour)
                candidates.append({
                    'x': x + w // 2,
                    'y': y + h // 2,
                    'area': area
                })
        
        if candidates:
            smallest = min(candidates, key=lambda c: c['area'])
            return {'x': smallest['x'], 'y': smallest['y']}
        
        return None
    
    def _calculate_motion(self, frame):
        """Calcula la intensidad de movimiento en el frame"""
        if self.prev_frame is None:
            return 0
        
        diff = cv2.absdiff(frame, self.prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        motion = np.sum(gray_diff) / (frame.shape[0] * frame.shape[1])
        
        return motion
