import numpy as np

class MatchAnalyzer:
    def __init__(self):
        self.shots = []
        self.positions = []
        self.motion_history = []
        self.last_shuttlecock_pos = None
        self.shot_in_progress = False
        self.frames_since_last_shot = 0
        self.min_frames_between_shots = 15  # Mínimo ~0.5 segundos entre golpes
        self.shuttlecock_trajectory = []  # Guardar trayectoria para mejor análisis
        
    def analyze_frame(self, detections, frame_num):
        """Analiza un frame y extrae información del juego"""
        
        self.frames_since_last_shot += 1
        
        # Guardar posiciones de jugadores para mapa de calor
        for player in detections['players']:
            self.positions.append({
                'x': player['x'],
                'y': player['y'],
                'frame': frame_num
            })
        
        # Guardar historial de movimiento
        self.motion_history.append(detections['motion_intensity'])
        
        # Detectar golpes solo si ha pasado suficiente tiempo
        shuttlecock = detections['shuttlecock']
        if shuttlecock:
            # Guardar trayectoria del volante
            self.shuttlecock_trajectory.append({'x': shuttlecock['x'], 'y': shuttlecock['y'], 'frame': frame_num})
            
            # Mantener solo últimos 30 frames de trayectoria
            if len(self.shuttlecock_trajectory) > 30:
                self.shuttlecock_trajectory.pop(0)
            
            if self.frames_since_last_shot >= self.min_frames_between_shots:
                shot_type = self._classify_shot(shuttlecock, detections['motion_intensity'])
                if shot_type:
                    self.shots.append({
                        'type': shot_type,
                        'frame': frame_num,
                        'position': shuttlecock
                    })
                    self.frames_since_last_shot = 0
                    self.shuttlecock_trajectory = []  # Resetear trayectoria después de golpe
            
            self.last_shuttlecock_pos = shuttlecock
    
    def _classify_shot(self, shuttlecock, motion_intensity):
        """Clasifica el tipo de golpe basado en trayectoria y velocidad"""
        
        if self.last_shuttlecock_pos is None:
            return None
        
        # Calcular velocidad y dirección
        dx = shuttlecock['x'] - self.last_shuttlecock_pos['x']
        dy = shuttlecock['y'] - self.last_shuttlecock_pos['y']
        speed = np.sqrt(dx**2 + dy**2)
        
        # Umbral mínimo más alto para considerar un golpe real
        if speed < 15:
            return None  # No hay golpe significativo
        
        # Analizar trayectoria completa para net shots
        is_net_shot = self._is_net_shot(shuttlecock, speed)
        if is_net_shot:
            return 'net'
        
        # Remate: MUY alta velocidad hacia abajo con ángulo pronunciado
        # Smash es el golpe más rápido y agresivo
        if speed > 50 and dy > 25 and dy > abs(dx) * 0.7:
            return 'smash'
        
        # Drop: velocidad media hacia abajo (más lento que smash)
        if 15 < speed < 40 and dy > 10 and dy > abs(dx) * 0.5:
            return 'drop'
        
        # Clear/Lob: hacia arriba con velocidad
        if dy < -15 and speed > 20:
            return 'clear'
        
        # Drive: horizontal rápido
        if abs(dx) > abs(dy) * 1.5 and speed > 25:
            return 'drive'
        
        return 'other'
    
    def _is_net_shot(self, current_pos, speed):
        """Detecta si es un net shot basado en trayectoria y posición"""
        if len(self.shuttlecock_trajectory) < 3:
            return False
        
        # Net shot características:
        # 1. Velocidad baja (golpe suave) 15-28
        # 2. Movimiento corto y controlado
        # 3. Se mantiene en la zona de red (parte superior de la cancha)
        # 4. Trayectoria descendente muy suave o casi horizontal
        
        # Velocidad suave característica de net shot
        if not (15 < speed < 28):
            return False
        
        # Verificar si está en zona de red (tercio superior de la imagen)
        # Ajustar según la perspectiva del video
        h = 360  # Resolución reducida
        in_net_zone = current_pos['y'] < h * 0.45  # Zona superior/frontal
        
        if not in_net_zone:
            return False
        
        # Analizar trayectoria reciente (últimos frames)
        if len(self.shuttlecock_trajectory) >= 5:
            recent = self.shuttlecock_trajectory[-5:]
            
            # Calcular distancia total recorrida
            total_distance = 0
            for i in range(1, len(recent)):
                dx = recent[i]['x'] - recent[i-1]['x']
                dy = recent[i]['y'] - recent[i-1]['y']
                dist = np.sqrt(dx**2 + dy**2)
                total_distance += dist
            
            avg_distance = total_distance / (len(recent) - 1)
            
            # Net shot: movimiento muy corto por frame (golpe contenido)
            if avg_distance < 25:
                return True
        
        # Detectar net shot por movimiento descendente muy suave
        if len(self.shuttlecock_trajectory) >= 3:
            recent = self.shuttlecock_trajectory[-3:]
            
            # Movimiento vertical
            vertical_movement = recent[-1]['y'] - recent[0]['y']
            horizontal_movement = abs(recent[-1]['x'] - recent[0]['x'])
            
            # Net shot: baja suavemente o se mantiene horizontal
            # Movimiento vertical pequeño (3-25 pixels) en zona de red
            if 3 < vertical_movement < 25 and in_net_zone:
                # Verificar que no sea muy horizontal (eso sería drive)
                if horizontal_movement < 40:
                    return True
            
            # También detectar net shot casi horizontal (golpe de toque)
            if abs(vertical_movement) < 15 and horizontal_movement < 35 and in_net_zone:
                return True
        
        return False
    
    def get_statistics(self):
        """Genera estadísticas del partido"""
        
        # Contar tipos de golpes
        shot_counts = {}
        for shot in self.shots:
            shot_type = shot['type']
            shot_counts[shot_type] = shot_counts.get(shot_type, 0) + 1
        
        # Calcular errores no forzados (golpes seguidos de baja actividad)
        unforced_errors = self._estimate_errors()
        
        # Clasificar ataque vs defensa
        attack_defense = self._classify_attack_defense()
        
        # Preparar datos para mapa de calor
        heatmap_data = self._prepare_heatmap()
        
        return {
            'total_shots': len(self.shots),
            'shot_types': shot_counts,
            'unforced_errors': unforced_errors,
            'attack_percentage': attack_defense['attack'],
            'defense_percentage': attack_defense['defense'],
            'heatmap': heatmap_data,
            'positions': self.positions
        }
    
    def _estimate_errors(self):
        """Estima errores no forzados basado en patrones de juego"""
        errors = 0
        
        # Buscar golpes seguidos de caída brusca en actividad
        for i in range(len(self.shots) - 1):
            shot_frame = self.shots[i]['frame']
            next_shot_frame = self.shots[i + 1]['frame']
            
            # Si hay mucho tiempo entre golpes, posible error
            if next_shot_frame - shot_frame > 100:  # ~2 segundos a 30fps
                errors += 1
        
        return errors
    
    def _classify_attack_defense(self):
        """Clasifica golpes en ataque o defensa"""
        attack_shots = ['smash', 'drop', 'net']
        defense_shots = ['clear', 'drive']
        
        attack_count = sum(1 for shot in self.shots if shot['type'] in attack_shots)
        defense_count = sum(1 for shot in self.shots if shot['type'] in defense_shots)
        total = attack_count + defense_count
        
        if total == 0:
            return {'attack': 0, 'defense': 0}
        
        return {
            'attack': round(attack_count / total * 100, 1),
            'defense': round(defense_count / total * 100, 1)
        }
    
    def _prepare_heatmap(self):
        """Prepara datos para el mapa de calor"""
        if not self.positions:
            return None
        
        # Crear matriz de densidad con resolución reducida
        heatmap = np.zeros((360, 640))
        
        # Agrupar posiciones cercanas para evitar ruido
        for pos in self.positions:
            x, y = int(pos['x']), int(pos['y'])
            if 0 <= x < 640 and 0 <= y < 360:
                heatmap[y, x] += 1
        
        return heatmap
