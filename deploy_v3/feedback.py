import json
import os

FEEDBACK_FILE = 'feedback_history.json'

def load_feedback():
    """Carga el historial de feedback"""
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_feedback(entry):
    """Guarda una corrección del usuario"""
    history = load_feedback()
    history.append(entry)
    
    # Mantener solo los últimos 20 feedbacks
    if len(history) > 20:
        history = history[-20:]
    
    with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def get_feedback_prompt():
    """Genera un prompt con ejemplos de correcciones anteriores"""
    history = load_feedback()
    
    if not history:
        return ""
    
    # Tomar los últimos 5 feedbacks como ejemplos
    recent = history[-5:]
    
    prompt = "\n\nAPRENDIZAJE DE CORRECCIONES ANTERIORES:\n"
    prompt += "El usuario ha corregido análisis previos. Usa estas correcciones para mejorar tu precisión:\n\n"
    
    for i, fb in enumerate(recent):
        prompt += f"Corrección {i+1}:\n"
        prompt += f"- Lo que detectaste: {json.dumps(fb.get('original', {}), ensure_ascii=False)}\n"
        prompt += f"- Lo correcto era: {json.dumps(fb.get('corrected', {}), ensure_ascii=False)}\n"
        if fb.get('comment'):
            prompt += f"- Comentario del usuario: {fb['comment']}\n"
        prompt += "\n"
    
    prompt += "Aplica estas correcciones a tu análisis actual. Si antes sobreestimabas o subestimabas algún tipo de golpe, ajusta tu criterio.\n"
    
    return prompt
