from flask import Flask, render_template, request, jsonify, send_from_directory, Response, session, redirect, url_for
from flask_cors import CORS
import os
import json
from functools import wraps
from werkzeug.utils import secure_filename
from analyzer_claude import ClaudeAnalyzer
from visualizer import Visualizer
from feedback import load_feedback, save_feedback
import threading
import time
import boto3
from datetime import datetime

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('SECRET_KEY', 'badminton-secret-2024')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Credenciales de acceso (configura via variables de entorno en Elastic Beanstalk)
# Para múltiples usuarios: APP_USERS=usuario1:pass1,usuario2:pass2,usuario3:pass3
def load_users():
    users_str = os.environ.get('APP_USERS', '')
    users = {}
    if users_str:
        for pair in users_str.split(','):
            if ':' in pair:
                username, password = pair.split(':', 1)
                users[username.strip()] = password.strip()
    # Fallback al usuario individual
    if not users:
        username = os.environ.get('APP_USERNAME', 'admin')
        password = os.environ.get('APP_PASSWORD', 'badminton2024')
        users[username] = password
    return users

USERS = load_users()

os.environ['AWS_ACCESS_KEY_ID'] = os.environ.get('AWS_ACCESS_KEY_ID', '')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ.get('AWS_SECRET_ACCESS_KEY', '')
os.environ['AWS_REGION'] = os.environ.get('AWS_REGION', 'us-east-1')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# S3 para almacenamiento de videos
S3_BUCKET = os.environ.get('S3_BUCKET', 'badminton-cdmx')
from botocore.config import Config
s3_client = boto3.client(
    's3',
    region_name='us-east-2',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    config=Config(signature_version='s3v4')
)

analysis_status = {}

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

analysis_status = {}

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if USERS.get(username) == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        error = 'Usuario o contraseña incorrectos'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/upload', methods=['POST'])
@login_required
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No se encontró el video'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Subir a S3 clasificado por usuario
    username = session.get('username', 'unknown')
    s3_key = f"{username}/{filename}"
    try:
        s3_client.upload_file(filepath, S3_BUCKET, s3_key)
        
        # Guardar metadata con el nombre seguro
        metadata_key = f"{username}/metadata/{filename}.json"
        metadata = {
            'filename': filename,
            'original_name': file.filename,
            'uploaded_at': datetime.now().isoformat(),
            'user': username,
            's3_key': s3_key
        }
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=metadata_key,
            Body=json.dumps(metadata),
            ContentType='application/json'
        )
    except Exception as e:
        print(f"ERROR S3 upload: {str(e)}")
    
    return jsonify({'success': True, 'filename': filename})

@app.route('/analyze/<filename>', methods=['POST'])
@login_required
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
        analyzer = ClaudeAnalyzer()
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], filename.split('.')[0])
        visualizer = Visualizer(output_dir)
        
        def update_progress(progress):
            analysis_status[analysis_id]['progress'] = progress
        
        # Analizar con Claude Vision
        stats = analyzer.analyze_video(filepath, progress_callback=update_progress)
        
        # Si hay error de Claude, reportarlo
        if 'error' in stats and stats.get('total_shots', 0) == 0:
            analysis_status[analysis_id]['status'] = 'error'
            analysis_status[analysis_id]['error'] = stats['error']
            return
        
        # Generar gráficos
        analysis_status[analysis_id]['progress'] = 95
        visualizer.generate_report(stats)
        
        # Limpiar video
        try:
            os.remove(filepath)
        except:
            pass
        
        result = {
            'success': True,
            'stats': {
                'total_shots': stats.get('total_shots', 0),
                'shot_types': stats.get('shot_types', {}),
                'unforced_errors': stats.get('unforced_errors', 0),
                'attack_percentage': stats.get('attack_percentage', 0),
                'defense_percentage': stats.get('defense_percentage', 0),
                'score_detected': stats.get('score_detected'),
                'observations': stats.get('observations', ''),
                'match_stats': stats.get('match_stats', {}),
                'player1': stats.get('player1', {}),
                'player2': stats.get('player2', {})
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
@login_required
def get_status(analysis_id):
    if analysis_id not in analysis_status:
        return jsonify({'error': 'Análisis no encontrado'}), 404
    
    return jsonify(analysis_status[analysis_id])

@app.route('/output/<path:filename>')
@login_required
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/feedback', methods=['POST'])
@login_required
def submit_feedback():
    data = request.json
    if not data:
        return jsonify({'error': 'No se recibieron datos'}), 400
    
    save_feedback({
        'original': data.get('original', {}),
        'corrected': data.get('corrected', {}),
        'comment': data.get('comment', '')
    })
    
    return jsonify({'success': True, 'message': 'Feedback guardado. Los próximos análisis serán más precisos.'})

@app.route('/feedback', methods=['GET'])
def get_feedback():
    history = load_feedback()
    return jsonify(history)

@app.route('/library')
@login_required
def library():
    """Lista los videos del usuario actual"""
    username = session.get('username', 'unknown')
    videos = []
    
    try:
        # Listar metadata del usuario
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=f"{username}/metadata/"
        )
        
        for obj in response.get('Contents', []):
            metadata_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=obj['Key'])
            metadata = json.loads(metadata_obj['Body'].read().decode('utf-8'))
            videos.append(metadata)
        
        # Ordenar por fecha (más recientes primero)
        videos.sort(key=lambda x: x.get('uploaded_at', ''), reverse=True)
    except Exception as e:
        print(f"ERROR listing library: {str(e)}")
    
    return jsonify({'success': True, 'videos': videos})

@app.route('/library/download/<filename>')
@login_required
def download_video(filename):
    """Sirve el video directamente desde S3 a través del servidor"""
    username = session.get('username', 'unknown')
    
    try:
        metadata_key = f"{username}/metadata/{filename}.json"
        metadata_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=metadata_key)
        metadata = json.loads(metadata_obj['Body'].read().decode('utf-8'))
        s3_key = metadata.get('s3_key', f"{username}/{filename}")
        
        # Descargar el video de S3 y servirlo directamente
        s3_response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        
        from flask import Response
        return Response(
            s3_response['Body'].iter_chunks(1024 * 1024),
            content_type='video/mp4',
            headers={
                'Content-Disposition': f'inline; filename="{filename}"',
                'Content-Length': str(s3_response['ContentLength'])
            }
        )
    except Exception as e:
        print(f"ERROR download: {str(e)}")
        return jsonify({'error': f'Video no encontrado: {str(e)}'}), 404

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.json
    if not data or 'question' not in data or 'analysis' not in data:
        return jsonify({'error': 'Faltan datos'}), 400
    
    question = data['question']
    analysis = data['analysis']
    
    try:
        import boto3
        client = boto3.client(
            'bedrock-runtime',
            region_name=os.environ.get('AWS_REGION', 'us-east-1'),
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
        
        context = f"""Eres un asistente experto en badminton. El usuario acaba de analizar un video de un partido y estos son los resultados:

Estadísticas del partido:
- Total de golpes: {analysis.get('total_shots', 0)}
- Tipos de golpes: {json.dumps(analysis.get('shot_types', {}), ensure_ascii=False)}
- Errores no forzados: {analysis.get('unforced_errors', 0)}
- Ataque general: {analysis.get('attack_percentage', 0)}%
- Defensa general: {analysis.get('defense_percentage', 0)}%
- Marcador detectado: {analysis.get('score_detected', 'No detectado')}
- Observaciones: {analysis.get('observations', '')}

Jugador 1 (parte inferior de la cancha):
- Golpes: {analysis.get('player1', {}).get('shots', 0)}
- Tipos: {json.dumps(analysis.get('player1', {}).get('shot_types', {}), ensure_ascii=False)}
- Errores NF: {analysis.get('player1', {}).get('unforced_errors', 0)}
- Ataque: {analysis.get('player1', {}).get('attack_percentage', 0)}%

Jugador 2 (parte superior de la cancha):
- Golpes: {analysis.get('player2', {}).get('shots', 0)}
- Tipos: {json.dumps(analysis.get('player2', {}).get('shot_types', {}), ensure_ascii=False)}
- Errores NF: {analysis.get('player2', {}).get('unforced_errors', 0)}
- Ataque: {analysis.get('player2', {}).get('attack_percentage', 0)}%

Responde la pregunta del usuario de forma breve y útil en español. Da consejos tácticos si aplica."""
        
        response = client.invoke_model(
            modelId='us.anthropic.claude-sonnet-4-6',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "messages": [
                    {"role": "user", "content": f"{context}\n\nPregunta del usuario: {question}"}
                ]
            }),
            contentType="application/json"
        )
        
        result = json.loads(response['body'].read())
        answer = result['content'][0]['text']
        
        return jsonify({'success': True, 'answer': answer})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
