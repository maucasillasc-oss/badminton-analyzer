import boto3
import json
import base64
import cv2
import numpy as np
import os
from shot_detector import ShotDetector
from feedback import get_feedback_prompt

class ClaudeAnalyzer:
    def __init__(self):
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=os.environ.get('AWS_REGION', 'us-east-1'),
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
        self.model_id = 'us.anthropic.claude-sonnet-4-6'
        self.detector = ShotDetector()
    
    def analyze_video(self, video_path, progress_callback=None):
        """Analiza video con YOLO + Claude"""
        
        if progress_callback:
            progress_callback(5)
        
        # Verificar que sea badminton
        is_badminton = self._verify_badminton(video_path)
        if not is_badminton:
            return {'error': 'Lo siento, solo puedo analizar videos de badminton'}
        
        if progress_callback:
            progress_callback(10)
        
        # Paso 1: YOLO + tracking detecta golpes
        detection = self.detector.process_video(video_path, progress_callback)
        
        if progress_callback:
            progress_callback(75)
        
        # Paso 2: Construir estadísticas
        stats = self._build_stats(detection)
        
        if progress_callback:
            progress_callback(80)
        
        # Paso 3: Claude genera análisis táctico
        observations = self._get_tactical_analysis(stats)
        stats['observations'] = observations
        
        if progress_callback:
            progress_callback(95)
        
        return stats
    
    def _build_stats(self, detection):
        """Construye estadísticas a partir de los golpes detectados"""
        shots = detection['shots']
        rallies = detection['rallies']
        
        # Separar por jugador
        p1_shots = [s for s in shots if s['player'] == 'J1']
        p2_shots = [s for s in shots if s['player'] == 'J2']
        
        # Contar tipos
        def count_types(player_shots):
            types = {}
            for s in player_shots:
                t = s['type']
                types[t] = types.get(t, 0) + 1
            return {k: v for k, v in types.items() if v > 0}
        
        p1_types = count_types(p1_shots)
        p2_types = count_types(p2_shots)
        
        # Combinados
        combined_types = {}
        for k, v in p1_types.items():
            combined_types[k] = combined_types.get(k, 0) + v
        for k, v in p2_types.items():
            combined_types[k] = combined_types.get(k, 0) + v
        
        # Ataque/defensa
        attack_types = {'smash', 'drop', 'net'}
        p1_attack = sum(1 for s in p1_shots if s['type'] in attack_types)
        p1_defense = len(p1_shots) - p1_attack
        p2_attack = sum(1 for s in p2_shots if s['type'] in attack_types)
        p2_defense = len(p2_shots) - p2_attack
        
        p1_tc = p1_attack + p1_defense
        p2_tc = p2_attack + p2_defense
        p1_attack_pct = round((p1_attack / p1_tc) * 100, 1) if p1_tc > 0 else 50
        p2_attack_pct = round((p2_attack / p2_tc) * 100, 1) if p2_tc > 0 else 50
        
        total_attack = p1_attack + p2_attack
        total_defense = p1_defense + p2_defense
        ttc = total_attack + total_defense
        attack_pct = round((total_attack / ttc) * 100, 1) if ttc > 0 else 50
        
        # Rally stats
        rally_lengths = [r['length'] for r in rallies if r['length'] > 0]
        avg_rally = round(sum(rally_lengths) / len(rally_lengths), 1) if rally_lengths else 0
        max_rally = max(rally_lengths) if rally_lengths else 0
        
        # Zonas
        def get_zones(player_shots):
            zones = {'front_left': 0, 'front_center': 0, 'front_right': 0, 'back_left': 0, 'back_center': 0, 'back_right': 0}
            total = len(player_shots)
            if total == 0:
                return zones
            for s in player_shots:
                z = s.get('zone', 'back_center')
                if z in zones:
                    zones[z] += 1
            # Convertir a porcentajes
            return {k: round((v / total) * 100) for k, v in zones.items()}
        
        # Saques (primer golpe de cada rally es serve)
        p1_serves = sum(1 for i, s in enumerate(shots) if s['player'] == 'J1' and (i == 0 or shots[i-1]['frame'] < s['frame'] - detection['fps'] * 3))
        p2_serves = sum(1 for i, s in enumerate(shots) if s['player'] == 'J2' and (i == 0 or shots[i-1]['frame'] < s['frame'] - detection['fps'] * 3))
        
        return {
            'total_shots': len(shots),
            'shot_types': combined_types,
            'unforced_errors': 0,  # No se puede detectar solo con tracking
            'attack_percentage': attack_pct,
            'defense_percentage': round(100 - attack_pct, 1),
            'score_detected': None,
            'observations': '',
            'match_stats': {
                'total_rallies': len(rally_lengths),
                'avg_rally_length': avg_rally,
                'max_rally_length': max_rally,
                'long_rallies_won': {'J1': 0, 'J2': 0},
                'short_rallies_won': {'J1': 0, 'J2': 0}
            },
            'player1': {
                'shots': len(p1_shots),
                'shot_types': p1_types,
                'winners': 0,
                'unforced_errors': 0,
                'attack_percentage': p1_attack_pct,
                'defense_percentage': round(100 - p1_attack_pct, 1),
                'we_ratio': 0,
                'points_won_on_serve': p1_serves,
                'shot_zones': get_zones(p1_shots)
            },
            'player2': {
                'shots': len(p2_shots),
                'shot_types': p2_types,
                'winners': 0,
                'unforced_errors': 0,
                'attack_percentage': p2_attack_pct,
                'defense_percentage': round(100 - p2_attack_pct, 1),
                'we_ratio': 0,
                'points_won_on_serve': p2_serves,
                'shot_zones': get_zones(p2_shots)
            }
        }
    
    def _get_tactical_analysis(self, stats):
        """Claude genera análisis táctico"""
        try:
            prompt = f"""Basado en estas estadísticas de un partido de badminton, genera un análisis táctico breve (2 oraciones en español):

{json.dumps(stats, ensure_ascii=False, indent=2)}

Enfócate en quién dominó y una recomendación táctica."""

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}]
                }),
                contentType="application/json"
            )
            result = json.loads(response['body'].read())
            return result['content'][0]['text']
        except Exception as e:
            print(f"ERROR Claude: {str(e)}")
            return ''
    
    def _verify_badminton(self, video_path):
        """Verifica que sea badminton"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return False
        
        positions = [int(total_frames * p) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
        images = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                small = cv2.resize(frame, (640, 360))
                _, buffer = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                b64 = base64.b64encode(buffer).decode('utf-8')
                images.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
        cap.release()
        
        if not images:
            return False
        try:
            content = images + [{"type": "text", "text": "¿Estas imágenes son de un partido de badminton? Responde SOLO 'si' o 'no'."}]
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 10, "messages": [{"role": "user", "content": content}]}),
                contentType="application/json"
            )
            result = json.loads(response['body'].read())
            answer = result['content'][0]['text'].strip().lower()
            return 'si' in answer or 'sí' in answer
        except:
            return True
