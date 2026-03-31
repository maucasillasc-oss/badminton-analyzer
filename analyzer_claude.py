import boto3
import json
import base64
import cv2
import numpy as np
import os
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
    
    def analyze_video(self, video_path, progress_callback=None):
        """Analiza un video de badminton extrayendo frames clave"""
        
        # Extraer frames clave del video
        frames = self._extract_key_frames(video_path, progress_callback)
        
        if not frames:
            return {'error': 'No se pudieron extraer frames del video'}
        
        # Dividir frames en lotes y analizar cada uno
        if progress_callback:
            progress_callback(50)
        
        batch_size = 20
        batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]
        
        partial_results = []
        for idx, batch in enumerate(batches):
            if progress_callback:
                progress_callback(50 + int((idx / len(batches)) * 35))
            
            result = self._analyze_batch(batch, idx + 1, len(batches))
            partial_results.append(result)
        
        # Combinar resultados de todos los lotes
        if progress_callback:
            progress_callback(90)
        
        final = self._combine_results(partial_results)
        return final
    
    def _extract_key_frames(self, video_path, progress_callback=None):
        """Extrae frames cada 1 segundo para cubrir todo el video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            fps = 30
        
        # Extraer un frame cada 1 segundo
        interval = int(fps)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                small = cv2.resize(frame, (640, 360))
                _, buffer = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 65])
                b64 = base64.b64encode(buffer).decode('utf-8')
                
                timestamp = frame_count / fps
                frames.append({
                    'image': b64,
                    'timestamp': round(timestamp, 1),
                    'frame': frame_count
                })
            
            frame_count += 1
            
            if progress_callback and frame_count % 200 == 0:
                progress = int((frame_count / total_frames) * 45)
                progress_callback(progress)
        
        cap.release()
        return frames

    def _analyze_batch(self, frames, batch_num, total_batches):
        """Analiza un lote de frames con Claude"""
        
        content = []
        
        content.append({
            "type": "text",
            "text": f"Estas son imágenes consecutivas (1 por segundo) de un partido de badminton. Lote {batch_num} de {total_batches}. Analiza CADA frame cuidadosamente para identificar TODOS los golpes."
        })
        
        for i, frame in enumerate(frames):
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame['image']
                }
            })
            content.append({
                "type": "text",
                "text": f"Segundo {frame['timestamp']}s"
            })
        
        content.append({
            "type": "text",
            "text": """Analiza TODOS los golpes visibles en estas imágenes del partido de badminton.

Para cada golpe que identifiques, considera:
- La postura del jugador (brazo arriba = posible smash/clear, agachado en red = net shot)
- La posición del volante (alto = clear, cayendo rápido = smash, cerca de red = net/drop)
- El movimiento del jugador entre frames consecutivos
- Si el jugador está atacando o defendiendo

Tipos de golpes:
- smash: Remate agresivo, jugador salta o golpea con fuerza hacia abajo
- drop: Golpe suave que cae justo después de la red, engañando al rival
- net: Jugador está cerca de la red y deja el volante suavemente del otro lado de la red
- clear: Golpe alto y profundo hacia el fondo de la cancha contraria
- drive: Golpe horizontal rápido a media altura
- serve: Servicio/saque
- other: Cualquier otro golpe

Responde SOLO con este JSON exacto:
{
    "shots_detected": número total de golpes en este segmento,
    "shot_types": {
        "smash": cantidad,
        "drop": cantidad,
        "net": cantidad,
        "clear": cantidad,
        "drive": cantidad,
        "serve": cantidad,
        "other": cantidad
    },
    "unforced_errors": errores no forzados visibles (volante a la red sin presión, fuera de cancha),
    "attack_shots": cantidad de golpes ofensivos,
    "defense_shots": cantidad de golpes defensivos,
    "score_visible": "marcador si es visible en algún frame, o null",
    "observations": "descripción breve en español de lo que pasa en este segmento"
}

IMPORTANTE: Cuenta TODOS los golpes que veas. Cada vez que un jugador golpea el volante es un golpe. No subestimes la cantidad.""" + get_feedback_prompt()
        })
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2000,
                    "messages": [
                        {"role": "user", "content": content}
                    ]
                }),
                contentType="application/json"
            )
            
            result = json.loads(response['body'].read())
            text_response = result['content'][0]['text']
            
            clean = text_response.strip()
            if clean.startswith('```'):
                clean = clean.split('\n', 1)[1]
                clean = clean.rsplit('```', 1)[0]
            
            return json.loads(clean)
            
        except Exception as e:
            return {
                'shots_detected': 0,
                'shot_types': {},
                'unforced_errors': 0,
                'attack_shots': 0,
                'defense_shots': 0,
                'score_visible': None,
                'observations': f'Error: {str(e)}'
            }
    
    def _combine_results(self, results):
        """Combina resultados de múltiples lotes"""
        total_shots = 0
        combined_types = {}
        total_errors = 0
        total_attack = 0
        total_defense = 0
        score = None
        all_observations = []
        
        for r in results:
            total_shots += r.get('shots_detected', 0)
            total_errors += r.get('unforced_errors', 0)
            total_attack += r.get('attack_shots', 0)
            total_defense += r.get('defense_shots', 0)
            
            for shot_type, count in r.get('shot_types', {}).items():
                combined_types[shot_type] = combined_types.get(shot_type, 0) + count
            
            if r.get('score_visible') and r['score_visible'] != 'null':
                score = r['score_visible']
            
            if r.get('observations') and not r['observations'].startswith('Error'):
                all_observations.append(r['observations'])
        
        # Calcular porcentajes
        total_classified = total_attack + total_defense
        attack_pct = round((total_attack / total_classified) * 100, 1) if total_classified > 0 else 50
        defense_pct = round((total_defense / total_classified) * 100, 1) if total_classified > 0 else 50
        
        # Eliminar tipos con 0 golpes
        combined_types = {k: v for k, v in combined_types.items() if v > 0}
        
        return {
            'total_shots': total_shots,
            'shot_types': combined_types,
            'unforced_errors': total_errors,
            'attack_percentage': attack_pct,
            'defense_percentage': defense_pct,
            'score_detected': score,
            'observations': ' '.join(all_observations) if all_observations else 'No se pudieron generar observaciones.'
        }
