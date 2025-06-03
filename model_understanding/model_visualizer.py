#!/usr/bin/env python3
"""
Detectron2 Model Architecture Visualizer
Creates visual representations of the model structure and data flow.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from typing import Dict, List, Tuple
import networkx as nx
from collections import defaultdict

class ModelVisualizer:
    """Visualizes model architecture and data flow"""
    
    def __init__(self, analyzer_results=None):
        self.analyzer_results = analyzer_results
        self.colors = {
            'backbone': '#3498db',
            'fpn': '#e74c3c', 
            'proposal_generator': '#f39c12',
            'roi_heads': '#2ecc71',
            'stem': '#9b59b6',
            'conv': '#34495e',
            'linear': '#e67e22',
            'norm': '#95a5a6',
            'activation': '#f1c40f'
        }
    
    def plot_architecture_overview(self, figsize=(16, 12)):
        """Create a high-level overview of the model architecture"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define main components and their positions
        components = [
            {'name': 'Input Image', 'pos': (1, 8), 'size': (1.5, 1), 'color': '#ecf0f1'},
            {'name': 'Backbone\n(ResNet-50)', 'pos': (1, 6), 'size': (1.5, 1.5), 'color': self.colors['backbone']},
            {'name': 'FPN', 'pos': (4, 6), 'size': (1.5, 1.5), 'color': self.colors['fpn']},
            {'name': 'RPN', 'pos': (7, 7), 'size': (1.5, 1), 'color': self.colors['proposal_generator']},
            {'name': 'ROI Heads', 'pos': (7, 5), 'size': (1.5, 1), 'color': self.colors['roi_heads']},
            {'name': 'Box Head', 'pos': (10, 6), 'size': (1.2, 0.8), 'color': self.colors['roi_heads']},
            {'name': 'Mask Head', 'pos': (10, 4.5), 'size': (1.2, 0.8), 'color': self.colors['roi_heads']},
            {'name': 'Predictions', 'pos': (13, 5.25), 'size': (1.5, 1.5), 'color': '#ecf0f1'}
        ]
        
        # Draw components
        boxes = {}
        for comp in components:
            rect = FancyBboxPatch(
                comp['pos'], comp['size'][0], comp['size'][1],
                boxstyle="round,pad=0.1",
                facecolor=comp['color'],
                edgecolor='black',
                linewidth=1.5,
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add text
            ax.text(
                comp['pos'][0] + comp['size'][0]/2,
                comp['pos'][1] + comp['size'][1]/2,
                comp['name'],
                ha='center', va='center',
                fontsize=10, fontweight='bold'
            )
            
            boxes[comp['name']] = comp
        
        # Draw connections
        connections = [
            ('Input Image', 'Backbone\n(ResNet-50)'),
            ('Backbone\n(ResNet-50)', 'FPN'),
            ('FPN', 'RPN'),
            ('FPN', 'ROI Heads'),
            ('RPN', 'ROI Heads'),
            ('ROI Heads', 'Box Head'),
            ('ROI Heads', 'Mask Head'),
            ('Box Head', 'Predictions'),
            ('Mask Head', 'Predictions')
        ]
        
        for start, end in connections:
            start_box = boxes[start]
            end_box = boxes[end]
            
            start_point = (
                start_box['pos'][0] + start_box['size'][0],
                start_box['pos'][1] + start_box['size'][1]/2
            )
            end_point = (
                end_box['pos'][0],
                end_box['pos'][1] + end_box['size'][1]/2
            )
            
            arrow = ConnectionPatch(
                start_point, end_point, "data", "data",
                arrowstyle="->", shrinkA=5, shrinkB=5,
                mutation_scale=20, fc="black", lw=2
            )
            ax.add_patch(arrow)
        
        ax.set_xlim(0, 15)
        ax.set_ylim(3, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Detectron2 Mask R-CNN Architecture Overview', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_backbone_details(self, figsize=(14, 10)):
        """Detailed view of the backbone architecture"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # ResNet-50 stages
        stages = [
            {'name': 'Input\n(3×H×W)', 'pos': (1, 8), 'channels': 3, 'size': 'H×W'},
            {'name': 'Stem\n7×7 conv', 'pos': (3, 8), 'channels': 64, 'size': 'H/2×W/2'},
            {'name': 'Res2\n3 blocks', 'pos': (5, 8), 'channels': 256, 'size': 'H/4×W/4'},
            {'name': 'Res3\n4 blocks', 'pos': (7, 8), 'channels': 512, 'size': 'H/8×W/8'},
            {'name': 'Res4\n6 blocks', 'pos': (9, 8), 'channels': 1024, 'size': 'H/16×W/16'},
            {'name': 'Res5\n3 blocks', 'pos': (11, 8), 'channels': 2048, 'size': 'H/32×W/32'}
        ]
        
        # FPN components
        fpn_components = [
            {'name': 'P2', 'pos': (5, 5), 'channels': 256, 'size': 'H/4×W/4'},
            {'name': 'P3', 'pos': (7, 5), 'channels': 256, 'size': 'H/8×W/8'},
            {'name': 'P4', 'pos': (9, 5), 'channels': 256, 'size': 'H/16×W/16'},
            {'name': 'P5', 'pos': (11, 5), 'channels': 256, 'size': 'H/32×W/32'},
            {'name': 'P6', 'pos': (13, 5), 'channels': 256, 'size': 'H/64×W/64'}
        ]
        
        # Draw backbone stages
        for i, stage in enumerate(stages):
            color_intensity = 0.3 + (i * 0.1)
            color = plt.cm.Blues(color_intensity)
            
            rect = FancyBboxPatch(
                stage['pos'], 1.5, 1.5,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='black',
                linewidth=1.5
            )
            ax.add_patch(rect)
            
            ax.text(
                stage['pos'][0] + 0.75, stage['pos'][1] + 1.1,
                stage['name'],
                ha='center', va='center',
                fontsize=9, fontweight='bold'
            )
            ax.text(
                stage['pos'][0] + 0.75, stage['pos'][1] + 0.4,
                f"{stage['channels']} ch\n{stage['size']}",
                ha='center', va='center',
                fontsize=8
            )
        
        # Draw FPN components
        for fpn in fpn_components:
            rect = FancyBboxPatch(
                fpn['pos'], 1.5, 1,
                boxstyle="round,pad=0.1",
                facecolor=self.colors['fpn'],
                edgecolor='black',
                linewidth=1.5,
                alpha=0.8
            )
            ax.add_patch(rect)
            
            ax.text(
                fpn['pos'][0] + 0.75, fpn['pos'][1] + 0.7,
                fpn['name'],
                ha='center', va='center',
                fontsize=10, fontweight='bold', color='white'
            )
            ax.text(
                fpn['pos'][0] + 0.75, fpn['pos'][1] + 0.3,
                f"{fpn['channels']} ch",
                ha='center', va='center',
                fontsize=8, color='white'
            )
        
        # Draw connections from backbone to FPN
        backbone_to_fpn = [(5, 8), (7, 8), (9, 8), (11, 8)]  # Res2-5 positions
        fpn_positions = [(5, 5), (7, 5), (9, 5), (11, 5)]     # P2-5 positions
        
        for (bx, by), (fx, fy) in zip(backbone_to_fpn, fpn_positions):
            arrow = ConnectionPatch(
                (bx + 0.75, by), (fx + 0.75, fy + 1),
                "data", "data",
                arrowstyle="->", shrinkA=5, shrinkB=5,
                mutation_scale=15, fc=self.colors['fpn'], lw=2
            )
            ax.add_patch(arrow)
        
        # Draw lateral connections in FPN
        for i in range(len(fpn_positions) - 1):
            start = (fpn_positions[i+1][0] + 0.75, fpn_positions[i+1][1] + 0.5)
            end = (fpn_positions[i][0] + 0.75, fpn_positions[i][1] + 0.5)
            
            arrow = ConnectionPatch(
                start, end, "data", "data",
                arrowstyle="->", shrinkA=5, shrinkB=5,
                mutation_scale=15, fc='orange', lw=2, alpha=0.7
            )
            ax.add_patch(arrow)
        
        # Add P6 connection
        arrow = ConnectionPatch(
            (11 + 0.75, 5 + 0.5), (13 + 0.75, 5 + 0.5),
            "data", "data",
            arrowstyle="->", shrinkA=5, shrinkB=5,
            mutation_scale=15, fc='red', lw=2
        )
        ax.add_patch(arrow)
        
        # Add labels
        ax.text(7, 10, 'ResNet-50 Backbone', ha='center', fontsize=14, fontweight='bold')
        ax.text(7, 3.5, 'Feature Pyramid Network (FPN)', ha='center', fontsize=14, fontweight='bold')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=plt.cm.Blues(0.6), label='Backbone Stages'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['fpn'], label='FPN Features'),
            plt.Line2D([0], [0], color='orange', lw=2, label='Lateral Connections'),
            plt.Line2D([0], [0], color='red', lw=2, label='Max Pool (P6)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_xlim(0, 15)
        ax.set_ylim(2, 11)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Backbone + FPN Detailed Architecture', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_data_flow(self, analysis_results, figsize=(16, 10)):
        """Plot data flow with tensor shapes through the network"""
        if not analysis_results:
            print("No analysis results provided. Run analyzer.analyze_forward_pass() first.")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract layer information
        layers = analysis_results.get('layer_outputs', {})
        y_pos = len(layers)
        
        for i, (layer_name, layer_data) in enumerate(layers.items()):
            y = y_pos - i
            
            # Determine color based on layer type
            layer_type = layer_data.get('layer_type', '')
            if 'backbone' in layer_name.lower():
                color = self.colors['backbone']
            elif 'fpn' in layer_name.lower():
                color = self.colors['fpn']
            elif 'rpn' in layer_name.lower() or 'proposal' in layer_name.lower():
                color = self.colors['proposal_generator']
            elif 'roi' in layer_name.lower():
                color = self.colors['roi_heads']
            else:
                color = self.colors['conv']
            
            # Draw layer box
            rect = FancyBboxPatch(
                (1, y-0.4), 6, 0.8,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor='black',
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Layer name
            ax.text(1.1, y, layer_name.split('.')[-1], 
                   va='center', fontsize=8, fontweight='bold')
            
            # Input shapes
            input_shapes = layer_data.get('input_shapes', [])
            input_text = self._format_shapes(input_shapes)
            ax.text(8, y, f"Input: {input_text}", va='center', fontsize=7)
            
            # Output shapes  
            output_shapes = layer_data.get('output_shapes', [])
            output_text = self._format_shapes(output_shapes)
            ax.text(12, y, f"Output: {output_text}", va='center', fontsize=7)
            
            # Draw arrow to next layer
            if i < len(layers) - 1:
                arrow = ConnectionPatch(
                    (4, y-0.4), (4, y-0.6),
                    "data", "data",
                    arrowstyle="->", shrinkA=2, shrinkB=2,
                    mutation_scale=15, fc="black"
                )
                ax.add_patch(arrow)
        
        ax.set_xlim(0, 18)
        ax.set_ylim(0, len(layers) + 1)
        ax.set_title('Data Flow Through Model Layers', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def _format_shapes(self, shapes):
        """Format tensor shapes for display"""
        if not shapes:
            return "None"
        
        formatted = []
        for shape in shapes[:2]:  # Show first 2 shapes only
            if isinstance(shape, list) and len(shape) > 0:
                if len(shape) == 4:  # Typical conv feature map [B, C, H, W]
                    formatted.append(f"[{shape[1]}×{shape[2]}×{shape[3]}]")
                elif len(shape) == 2:  # Typical linear layer [B, Features]
                    formatted.append(f"[{shape[1]}]")
                else:
                    formatted.append(str(shape))
            else:
                formatted.append(str(shape)[:20])
        
        if len(shapes) > 2:
            formatted.append("...")
        
        return ", ".join(formatted)
    
    def create_summary_report(self, analyzer, analysis_results=None, save_path="model_report.png"):
        """Create a comprehensive visual report"""
        fig = plt.figure(figsize=(20, 16))
        
        # Architecture overview
        ax1 = plt.subplot(3, 2, (1, 2))
        self.plot_architecture_overview()
        plt.sca(ax1)
        
        # Backbone details
        ax2 = plt.subplot(3, 2, (3, 4))
        self.plot_backbone_details()
        plt.sca(ax2)
        
        # Parameter distribution
        ax3 = plt.subplot(3, 2, 5)
        self._plot_parameter_distribution(analyzer)
        
        # Layer count by type
        ax4 = plt.subplot(3, 2, 6)
        self._plot_layer_types(analyzer)
        
        plt.suptitle('Detectron2 Model Analysis Report', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Report saved to {save_path}")
        return fig
    
    def _plot_parameter_distribution(self, analyzer):
        """Plot parameter distribution across components"""
        components = defaultdict(int)
        
        for name, info in analyzer.layer_info.items():
            component = name.split('.')[0] if '.' in name else 'other'
            components[component] += info['parameters']['total']
        
        # Filter out components with no parameters
        components = {k: v for k, v in components.items() if v > 0}
        
        plt.pie(components.values(), labels=components.keys(), autopct='%1.1f%%')
        plt.title('Parameter Distribution by Component')
    
    def _plot_layer_types(self, analyzer):
        """Plot distribution of layer types"""
        layer_types = defaultdict(int)
        
        for info in analyzer.layer_info.values():
            layer_types[info['type']] += 1
        
        # Get top 10 most common layer types
        top_types = sorted(layer_types.items(), key=lambda x: x[1], reverse=True)[:10]
        
        types, counts = zip(*top_types)
        plt.barh(range(len(types)), counts)
        plt.yticks(range(len(types)), types)
        plt.xlabel('Count')
        plt.title('Layer Types Distribution')
        plt.gca().invert_yaxis()

def main():
    """Example usage"""
    # This would typically be called after running the ModelAnalyzer
    visualizer = ModelVisualizer()
    
    # Create architecture overview
    fig1 = visualizer.plot_architecture_overview()
    plt.show()
    
    # Create backbone details
    fig2 = visualizer.plot_backbone_details()
    plt.show()

if __name__ == "__main__":
    main()