import os
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para threads
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import json

class Visualizer:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(self, stats):
        """Genera reporte completo con gráficos y estadísticas"""
        
        # Guardar estadísticas en JSON
        self._save_json(stats)
        
        # Generar gráfico de tipos de golpes
        if stats.get('shot_types'):
            self._plot_shot_types(stats['shot_types'])
        
        # Generar gráfico de ataque vs defensa
        if stats.get('attack_percentage') and stats.get('defense_percentage'):
            self._plot_attack_defense(stats)
        
        # Generar resumen en texto
        self._generate_text_summary(stats)
        
        print(f"\n✓ Gráficos guardados en '{self.output_dir}'")
    
    def _save_json(self, stats):
        """Guarda estadísticas en formato JSON"""
        output = {
            'total_shots': stats.get('total_shots', 0),
            'shot_types': stats.get('shot_types', {}),
            'unforced_errors': stats.get('unforced_errors', 0),
            'attack_percentage': stats.get('attack_percentage', 0),
            'defense_percentage': stats.get('defense_percentage', 0)
        }
        
        with open(os.path.join(self.output_dir, 'stats.json'), 'w') as f:
            json.dump(output, f, indent=2)
    
    def _plot_shot_types(self, shot_types):
        """Genera gráfico de barras con tipos de golpes"""
        if not shot_types:
            return
        
        plt.figure(figsize=(10, 6))
        
        types = list(shot_types.keys())
        counts = list(shot_types.values())
        
        # Colores: azul rey, verde-amarillo, rojo
        colors = ['#0047AB', '#00CED1', '#7FFF00', '#FFD700', '#FF6347', '#DC143C']
        plt.bar(types, counts, color=colors[:len(types)])
        
        plt.title('Distribución de Tipos de Golpes', fontsize=16, fontweight='bold')
        plt.xlabel('Tipo de Golpe', fontsize=12)
        plt.ylabel('Cantidad', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, 'shot_types.png'), dpi=300)
        plt.close()
    
    def _plot_attack_defense(self, stats):
        """Genera gráfico circular de ataque vs defensa"""
        plt.figure(figsize=(8, 8))
        
        sizes = [stats['attack_percentage'], stats['defense_percentage']]
        labels = ['Ataque', 'Defensa']
        colors = ['#DC143C', '#0047AB']  # Rojo para ataque, azul rey para defensa
        explode = (0.1, 0)
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        
        plt.title('Distribución Ataque vs Defensa', fontsize=16, fontweight='bold')
        plt.axis('equal')
        
        plt.savefig(os.path.join(self.output_dir, 'attack_defense.png'), dpi=300)
        plt.close()
    
    def _plot_heatmap(self, heatmap):
        """Genera mapa de calor de posiciones en cancha"""
        plt.figure(figsize=(14, 10))
        
        # Aplicar suavizado gaussiano más suave
        from scipy.ndimage import gaussian_filter
        heatmap_smooth = gaussian_filter(heatmap, sigma=30)
        
        # Normalizar para mejor visualización
        if heatmap_smooth.max() > 0:
            heatmap_smooth = heatmap_smooth / heatmap_smooth.max()
        
        # Crear colormap personalizado (azul frío a rojo caliente)
        colors = ['#000000', '#001a33', '#003366', '#0066cc', '#00ccff', 
                  '#00ff99', '#ffff00', '#ff9900', '#ff3300', '#cc0000']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # Mostrar solo la región de la cancha
        h, w = heatmap_smooth.shape
        court_region = heatmap_smooth[int(h*0.2):int(h*0.9), int(w*0.1):int(w*0.9)]
        
        plt.imshow(court_region, cmap=cmap, aspect='auto', interpolation='bilinear')
        plt.colorbar(label='Densidad de Posiciones', shrink=0.8)
        
        plt.title('Mapa de Calor - Posiciones de Jugadores en Cancha', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Ancho de la Cancha', fontsize=12)
        plt.ylabel('Largo de la Cancha', fontsize=12)
        
        # Remover ticks para mejor visualización
        plt.xticks([])
        plt.yticks([])
        
        plt.savefig(os.path.join(self.output_dir, 'heatmap.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _generate_text_summary(self, stats):
        """Genera resumen en texto plano"""
        summary = f"""
╔══════════════════════════════════════════════════════════╗
║          ANÁLISIS DE PARTIDO DE BADMINTON               ║
╚══════════════════════════════════════════════════════════╝

📊 ESTADÍSTICAS GENERALES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total de golpes detectados: {stats.get('total_shots', 0)}
Errores no forzados estimados: {stats.get('unforced_errors', 0)}

🎯 TIPOS DE GOLPES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for shot_type, count in stats.get('shot_types', {}).items():
            percentage = (count / stats.get('total_shots', 1) * 100) if stats.get('total_shots', 0) > 0 else 0
            summary += f"  • {shot_type.capitalize():12} : {count:3} ({percentage:.1f}%)\n"
        
        summary += f"""
⚔️  ESTILO DE JUEGO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Ataque  : {stats.get('attack_percentage', 0)}%
  • Defensa : {stats.get('defense_percentage', 0)}%

📁 ARCHIVOS GENERADOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • stats.json - Datos en formato JSON
  • shot_types.png - Gráfico de tipos de golpes
  • attack_defense.png - Gráfico ataque vs defensa
  • heatmap.png - Mapa de calor de posiciones
  • summary.txt - Este resumen
"""
        
        with open(os.path.join(self.output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(summary)
