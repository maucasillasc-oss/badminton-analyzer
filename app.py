from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os
import json
from werkzeug.utils import secure_filename
from detector_improved import BadmintonDetectorImproved
from analyzer import MatchAnalyzer
from visualizer import Visualizer
import cv2
import threading
import time

app = Flask(__name__)
CORS(app)  # Habilitar CORS
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Estado del análisis
analysis_status = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No se encontró el video'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    return jsonify({'success': True, 'filename': filename})

@app.route('/analyze/<filename>', methods=['POST'])
def analyze_video(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Video no encontrado'}), 404
    
    # Iniciar análisis en segundo plano
    analysis_id = filename.split('.')[0]
    analysis_status[analysis_id] = {'progress': 0, 'status': 'processing', 'result': None}
    
    thread = threading.Thread(target=process_video, args=(filepath, filename, analysis_id))
    thread.start()
    
    return jsonify({'success': True, 'analysis_id': analysis_id})

def process_video(filepath, filename, analysis_id):
    try:
        # Inicializar componentes con detector mejorado (HOG)
        detector = BadmintonDetectorImproved()
        analyzer = MatchAnalyzer()
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], filename.split('.')[0])
        visualizer = Visualizer(output_dir)
        
        # Procesar video
        cap = cv2.VideoCapture(filepath)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % 5 == 0:
                detections = detector.detect(frame)
                analyzer.analyze_frame(detections, frame_count)
                
                # Actualizar progreso
                progress = int((frame_count / total_frames) * 80)  # 80% para procesamiento
                analysis_status[analysis_id]['progress'] = progress
            
            frame_count += 1
        
        cap.release()
        
        # Generar estadísticas (20% restante)
        analysis_status[analysis_id]['progress'] = 85
        stats = analyzer.get_statistics()
        
        analysis_status[analysis_id]['progress'] = 90
        visualizer.generate_report(stats)
        
        # Preparar respuesta
        result = {
            'success': True,
            'stats': {
                'total_shots': stats['total_shots'],
                'shot_types': stats['shot_types'],
                'unforced_errors': stats['unforced_errors'],
                'attack_percentage': stats['attack_percentage'],
                'defense_percentage': stats['defense_percentage']
            },
            'images': {
                'shot_types': f'/output/{filename.split(".")[0]}/shot_types.png',
                'attack_defense': f'/output/{filename.split(".")[0]}/attack_defense.png'
            }
        }
        
        analysis_status[analysis_id]['progress'] = 100
        analysis_status[analysis_id]['status'] = 'completed'
        analysis_status[analysis_id]['result'] = result
        
    except Exception as e:
        analysis_status[analysis_id]['status'] = 'error'
        analysis_status[analysis_id]['error'] = str(e)

@app.route('/status/<analysis_id>')
def get_status(analysis_id):
    if analysis_id not in analysis_status:
        return jsonify({'error': 'Análisis no encontrado'}), 404
    
    return jsonify(analysis_status[analysis_id])

@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
