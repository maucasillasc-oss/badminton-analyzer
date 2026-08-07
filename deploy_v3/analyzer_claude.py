"""
Badminton Video Analyzer
Uses the Hit-Frame Detection paper pipeline for accurate hit detection,
then Claude for tactical analysis and shot classification.
"""
import boto3
import json
import base64
import cv2
import numpy as np
import os
from hitframe_detector import HitFrameDetector
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
        self.detector = HitFrameDetector()
    
    def analyze_video(self, video_path, progress_callback=None):
        """
        Full analysis pipeline:
        1. Verify it's badminton
        2. Hit-frame detection (paper pipeline)
        3. Extract frames at hit points for Claude classification
        4. Claude classifies shot types and generates tactical analysis
        """
        if progress_callback:
            progress_callback(5)
        
        # Step 1: Verify it's a badminton video
        is_badminton = self._verify_badminton(video_path)
        if not is_badminton:
            return {'error': 'Lo siento, solo puedo analizar videos de badminton'}
        
        if progress_callback:
            progress_callback(10)
        
        # Step 2: Hit-frame detection with the paper's models
        detection = self.detector.process_video(video_path, progress_callback)
        
        if progress_callback:
            progress_callback(72)
        
        # Step 3: Extract hit frames and classify with Claude
        hit_frames_info = self._extract_hit_frames(video_path, detection)
        
        if progress_callback:
            progress_callback(80)
        
        # Step 4: Claude classifies shots and generates analysis
        stats = self._classify_and_build_stats(
            video_path, detection, hit_frames_info, progress_callback
        )
        
        if progress_callback:
            progress_callback(95)
        
        return stats
    
    def _extract_hit_frames(self, video_path, detection):
        """Extract frame images at detected hit points."""
        cap = cv2.VideoCapture(video_path)
        fps = detection['fps']
        frame_height = detection['frame_height']
        mid_y = frame_height // 2
        
        hit_frames_info = []
        
        for rally in detection['rallies']:
            for hit_frame_num in rally['hit_frames']:
                cap.set(cv2.CAP_PROP_POS_FRAMES, hit_frame_num)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Determine which player hit based on frame position in direction sequence
                # The hit frame is where direction changes, so the player who hit
                # is on the side the shuttle is NOW going away from
                hit_idx = rally['hit_frames'].index(hit_frame_num)
                directions = rally['directions']
                
                # Find the direction after this hit
                # Hit indices map to positions in the direction sequence
                # Direction 1 = flying up (bottom player hit it)
                # Direction 2 = flying down (top player hit it)
                player = 'J1'  # Default: bottom player
                
                # Get direction at this point in the rally
                rally_frame_idx = hit_idx
                if rally_frame_idx < len(directions):
                    dir_after = directions[rally_frame_idx]
                    if dir_after == 1:
                        player = 'J1'  # Shuttle going up -> bottom player (J1) hit it
                    elif dir_after == 2:
                        player = 'J2'  # Shuttle going down -> top player (J2) hit it
                
                hit_frames_info.append({
                    'frame_num': hit_frame_num,
                    'timestamp': round(hit_frame_num / fps, 1),
                    'player': player,
                    'rally_idx': detection['rallies'].index(rally)
                })
        
        cap.release()
        return hit_frames_info
    
    def _classify_and_build_stats(self, video_path, detection, hit_frames_info, progress_callback=None):
        """
        Use Claude to classify shot types from hit frame images,
        then build complete statistics.
        """
        total_hits = detection['total_hits']
        fps = detection['fps']
        
        # If we have hits, sample some frames for Claude to classify
        shot_types_all = {}
        p1_types = {}
        p2_types = {}
        
        if hit_frames_info and total_hits > 0:
            # Sample up to 10 hit frames for Claude classification
            sample_size = min(10, len(hit_frames_info))
            sample_indices = np.linspace(0, len(hit_frames_info) - 1, sample_size, dtype=int)
            sampled_hits = [hit_frames_info[i] for i in sample_indices]
            
            # Get Claude to classify these shots
            classifications = self._claude_classify_shots(video_path, sampled_hits)
            
            if classifications:
                # Extrapolate to all hits based on proportions
                type_proportions = {}
                for cls in classifications:
                    t = cls.get('type', 'other')
                    type_proportions[t] = type_proportions.get(t, 0) + 1
                
                total_classified = sum(type_proportions.values())
                
                # Apply proportions to all hits
                for shot_type, count in type_proportions.items():
                    proportion = count / total_classified
                    estimated = max(1, round(total_hits * proportion))
                    shot_types_all[shot_type] = estimated
                
                # Distribute by player
                p1_hits = [h for h in hit_frames_info if h['player'] == 'J1']
                p2_hits = [h for h in hit_frames_info if h['player'] == 'J2']
                
                for shot_type, count in type_proportions.items():
                    proportion = count / total_classified
                    p1_types[shot_type] = max(0, round(len(p1_hits) * proportion))
                    p2_types[shot_type] = max(0, round(len(p2_hits) * proportion))
            else:
                # Fallback: can't classify, mark all as 'other'
                shot_types_all = {'other': total_hits}
                p1_hits = [h for h in hit_frames_info if h['player'] == 'J1']
                p2_hits = [h for h in hit_frames_info if h['player'] == 'J2']
                p1_types = {'other': len(p1_hits)}
                p2_types = {'other': len(p2_hits)}
        else:
            p1_hits = []
            p2_hits = []
        
        # Adjust totals to match actual hit count
        type_sum = sum(shot_types_all.values())
        if type_sum > 0 and type_sum != total_hits:
            diff = total_hits - type_sum
            # Add/subtract from the largest category
            if shot_types_all:
                largest = max(shot_types_all, key=shot_types_all.get)
                shot_types_all[largest] = max(1, shot_types_all[largest] + diff)
        
        # Calculate attack/defense percentages
        attack_types = {'smash', 'drop', 'net'}
        p1_attack = sum(v for k, v in p1_types.items() if k in attack_types)
        p2_attack = sum(v for k, v in p2_types.items() if k in attack_types)
        p1_total = sum(p1_types.values()) or 1
        p2_total = sum(p2_types.values()) or 1
        
        p1_attack_pct = round((p1_attack / p1_total) * 100, 1)
        p2_attack_pct = round((p2_attack / p2_total) * 100, 1)
        total_attack = p1_attack + p2_attack
        total_total = p1_total + p2_total
        attack_pct = round((total_attack / total_total) * 100, 1)
        
        # Rally statistics
        rally_lengths = [r['num_hits'] for r in detection['rallies'] if r['num_hits'] > 0]
        avg_rally = round(sum(rally_lengths) / len(rally_lengths), 1) if rally_lengths else 0
        max_rally = max(rally_lengths) if rally_lengths else 0
        
        # Build stats dict
        stats = {
            'total_shots': total_hits,
            'shot_types': shot_types_all,
            'unforced_errors': 0,
            'attack_percentage': attack_pct,
            'defense_percentage': round(100 - attack_pct, 1),
            'score_detected': None,
            'observations': '',
            'match_stats': {
                'total_rallies': len(detection['rallies']),
                'avg_rally_length': avg_rally,
                'max_rally_length': max_rally,
                'long_rallies_won': {'J1': 0, 'J2': 0},
                'short_rallies_won': {'J1': 0, 'J2': 0}
            },
            'player1': {
                'shots': len(p1_hits) if hit_frames_info else total_hits // 2,
                'shot_types': p1_types,
                'winners': 0,
                'unforced_errors': 0,
                'attack_percentage': p1_attack_pct,
                'defense_percentage': round(100 - p1_attack_pct, 1),
                'we_ratio': 0,
                'points_won_on_serve': 0,
                'shot_zones': {}
            },
            'player2': {
                'shots': len(p2_hits) if hit_frames_info else total_hits // 2,
                'shot_types': p2_types,
                'winners': 0,
                'unforced_errors': 0,
                'attack_percentage': p2_attack_pct,
                'defense_percentage': round(100 - p2_attack_pct, 1),
                'we_ratio': 0,
                'points_won_on_serve': 0,
                'shot_zones': {}
            }
        }
        
        # Get tactical analysis from Claude
        if progress_callback:
            progress_callback(88)
        
        observations = self._get_tactical_analysis(stats)
        stats['observations'] = observations
        
        return stats
    
    def _claude_classify_shots(self, video_path, sampled_hits):
        """Send hit frame images to Claude to classify shot types."""
        cap = cv2.VideoCapture(video_path)
        images = []
        
        for hit in sampled_hits:
            cap.set(cv2.CAP_PROP_POS_FRAMES, hit['frame_num'])
            ret, frame = cap.read()
            if ret:
                small = cv2.resize(frame, (640, 360))
                _, buffer = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                b64 = base64.b64encode(buffer).decode('utf-8')
                images.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}
                })
        
        cap.release()
        
        if not images:
            return None
        
        try:
            prompt_text = f"""Analiza estas {len(images)} imágenes de un partido de badminton. 
Cada imagen corresponde al momento exacto de un golpe detectado.
Para cada imagen, clasifica el tipo de golpe como UNO de: smash, clear, drop, net, drive, serve.

Responde SOLO con un JSON array, ejemplo:
[{{"type": "smash"}}, {{"type": "clear"}}, {{"type": "net"}}]

Criterios:
- smash: golpe explosivo hacia abajo, brazo estirado arriba
- clear: golpe alto y profundo al fondo
- drop: golpe suave que cae cerca de la red
- net: golpe en la red, jugador agachado
- drive: golpe rápido horizontal a media altura
- serve: saque (primer golpe, posición lateral)"""

            content = images + [{"type": "text", "text": prompt_text}]
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 300,
                    "messages": [{"role": "user", "content": content}]
                }),
                contentType="application/json"
            )
            
            result = json.loads(response['body'].read())
            answer = result['content'][0]['text'].strip()
            
            # Parse JSON from response
            # Handle cases where Claude wraps in markdown
            if '```' in answer:
                answer = answer.split('```')[1]
                if answer.startswith('json'):
                    answer = answer[4:]
                answer = answer.strip()
            
            classifications = json.loads(answer)
            return classifications
            
        except Exception as e:
            print(f"Claude shot classification error: {e}")
            return None
    
    def _get_tactical_analysis(self, stats):
        """Claude generates tactical analysis from stats."""
        try:
            feedback_prompt = get_feedback_prompt()
            
            prompt = f"""Basado en estas estadísticas de un partido de badminton, genera un análisis táctico breve (3-4 oraciones en español):

Total de golpes detectados: {stats['total_shots']}
Tipos de golpes: {json.dumps(stats.get('shot_types', {}), ensure_ascii=False)}
Rallies totales: {stats['match_stats']['total_rallies']}
Rally promedio: {stats['match_stats']['avg_rally_length']} golpes
Rally más largo: {stats['match_stats']['max_rally_length']} golpes
J1 golpes: {stats['player1']['shots']}, ataque: {stats['player1']['attack_percentage']}%
J2 golpes: {stats['player2']['shots']}, ataque: {stats['player2']['attack_percentage']}%

{feedback_prompt}

Enfócate en: quién dominó, estilo de juego de cada jugador, y una recomendación táctica."""

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 300,
                    "messages": [{"role": "user", "content": prompt}]
                }),
                contentType="application/json"
            )
            result = json.loads(response['body'].read())
            return result['content'][0]['text']
        except Exception as e:
            print(f"Claude tactical analysis error: {e}")
            return ''
    
    def _verify_badminton(self, video_path):
        """Verify the video is actually a badminton match."""
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
                images.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}
                })
        cap.release()
        
        if not images:
            return False
        
        try:
            content = images + [{
                "type": "text",
                "text": "¿Estas imágenes son de un partido de badminton? Responde SOLO 'si' o 'no'."
            }]
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": content}]
                }),
                contentType="application/json"
            )
            result = json.loads(response['body'].read())
            answer = result['content'][0]['text'].strip().lower()
            return 'si' in answer or 'sí' in answer
        except:
            return True
