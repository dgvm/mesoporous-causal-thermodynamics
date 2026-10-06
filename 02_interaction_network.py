import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-paper')

OUTPUT_DIR = "."
DATA_FILE = "../../ml_pipeline/processed_data/universal_synthesis_data.csv"

def build_interaction_network():
    """
    Constructs a non-linear interaction network mapping the predictive relationships 
    between chemical descriptors, synthesis parameters, and final textural properties.
    """
    # 1. Load Preprocessed Data
    df = pd.read_csv(DATA_FILE)
    
    targets = ['SBET', 'DP', 'VT']
    features_chem = ['Surf_TPSA', 'Surf_LogP', 'Surf_MW', 'Estimated_pH']
    features_syn  = ['Surfactant_Mass', 'Tes', 'ts', 'Teh', 'th', 'ta']
    features = features_chem + features_syn
    
    df_clean = df.dropna(subset=targets + features)
    X = df_clean[features]
    Y = df_clean[targets]
    
    # 2. Extract Feature Importances via Random Forest
    importances = {}
    for target in targets:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, Y[target])
        importances[target] = rf.feature_importances_
        
    # 3. Build Network Graph
    G = nx.DiGraph()
    
    for t in targets:
        G.add_node(t, type='target')
    for f in features_chem:
        G.add_node(f, type='chem')
    for f in features_syn:
        G.add_node(f, type='syn')
        
    # Add edges if importance exceeds threshold (filter noise)
    for t in targets:
        for idx, f in enumerate(features):
            weight = importances[t][idx]
            if weight > 0.03: 
                G.add_edge(f, t, weight=weight)
                
    # 4. Custom Spatial Positions
    pos = {
        'SBET': (0, 1),
        'DP': (0, 0),
        'VT': (0, -1)
    }
    
    y_chem = np.linspace(1.5, -1.5, len(features_chem))
    for i, f in enumerate(features_chem):
        pos[f] = (-1, y_chem[i])
        
    y_syn = np.linspace(1.5, -1.5, len(features_syn))
    for i, f in enumerate(features_syn):
        pos[f] = (1, y_syn[i])
        
    # 5. Visualization Setup
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Normalize weights to scale edge thickness visually
    scaler = MinMaxScaler(feature_range=(1, 8))
    weights_scaled = scaler.fit_transform(np.array(weights).reshape(-1, 1)).flatten()
    
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, width=weights_scaled, alpha=0.5, 
                           edge_color='#555555', connectionstyle="arc3,rad=0.1", 
                           arrowsize=20, node_size=1500)
                           
    edge_labels = { (u, v): f"{G[u][v]['weight']*100:.1f}%" for u, v in edges }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, 
                                 font_color='#e41a1c', label_pos=0.3, ax=ax)
                                 
    # Draw Nodes based on category
    for node in G.nodes():
        if G.nodes[node]['type'] == 'target':
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color='#e41a1c', node_size=2000, node_shape='*', ax=ax, edgecolors='k')
        elif G.nodes[node]['type'] == 'chem':
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color='#377eb8', node_size=1000, node_shape='s', ax=ax, edgecolors='k')
        else:
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color='#4daf4a', node_size=1000, node_shape='o', ax=ax, edgecolors='k')
            
    # Standard Nomenclature Labels
    labels = {
        'SBET': r'$S_{BET}$', 'DP': r'$D_P$', 'VT': r'$V_T$',
        'Surf_TPSA': r'TPSA', 'Surf_LogP': r'$\log P$', 'Surf_MW': r'$M_w$', 'Estimated_pH': r'$\widetilde{pH}$',
        'Surfactant_Mass': r'Mass$_{surf}$', 'Tes': r'$T_{es}$', 'ts': r'$t_s$', 'Teh': r'$T_{eh}$', 'th': r'$t_h$', 'ta': r'$t_a$'
    }
    
    for node, (x, y) in pos.items():
        if G.nodes[node]['type'] == 'chem':
            ax.text(x - 0.15, y, labels[node], ha='right', va='center', fontsize=12, fontweight='bold', color='#377eb8')
        elif G.nodes[node]['type'] == 'syn':
            ax.text(x + 0.15, y, labels[node], ha='left', va='center', fontsize=12, fontweight='bold', color='#4daf4a')
        else:
            ax.text(x, y + 0.15, labels[node], ha='center', va='bottom', fontsize=14, fontweight='bold', color='#e41a1c',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0))
                    
    ax.axis('off')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='Chemical Descriptors', markerfacecolor='#377eb8', markersize=12, markeredgecolor='k'),
        Line2D([0], [0], marker='*', color='w', label='Target Properties', markerfacecolor='#e41a1c', markersize=18, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Synthesis Variables', markerfacecolor='#4daf4a', markersize=12, markeredgecolor='k'),
        Line2D([0], [0], color='gray', lw=4, label='Interaction Strength (RF Importance)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=11, frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig68_interaction_network.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    build_interaction_network()
