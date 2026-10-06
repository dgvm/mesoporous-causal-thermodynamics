import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.estimators import HillClimbSearch, BIC
import os
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-paper')

OUTPUT_DIR = "."
DATA_FILE = "../../ml_pipeline/processed_data/universal_synthesis_data.csv"

def build_causal_dag():
    """
    Constructs a Directed Acyclic Graph (DAG) using Bayesian structure learning 
    to infer causal pathways in mesoporous materials synthesis.
    """
    # 1. Load Preprocessed Data
    # Preprocessing includes feature engineering, handling missing values, and normalization.
    df = pd.read_csv(DATA_FILE)
    
    important_vars = [
        'Estimated_pH', 'Surf_TPSA', 'Surf_LogP', 'Tes', 'Teh', 'ts', 'th', 'ta',
        'Surfactant_Mass', 'H2O/TEOS_molar', 'SBET', 'DP', 'VT'
    ]
    df_dag = df[important_vars]
    
    # 2. Data Discretization for Bayesian Network
    df_disc = pd.DataFrame()
    for col in df_dag.columns:
        df_disc[col] = pd.qcut(df_dag[col], q=4, duplicates='drop', labels=False).astype(str)
        
    # 3. Structure Learning via Hill Climbing
    hc = HillClimbSearch(df_disc)
    best_model = hc.estimate(scoring_method=BIC(df_disc))
    
    # Filter non-physical reverse edges (textural targets cannot cause synthesis conditions)
    targets = ['SBET', 'DP', 'VT']
    edges = [(u, v) for u, v in best_model.edges() if u not in targets]
    
    # 4. Nomenclature Mapping
    rename_map = {
        'Estimated_pH': r'$\widetilde{pH}$',
        'Surf_TPSA': r'TPSA',
        'Surf_LogP': r'$\log P$',
        'Surf_MW': r'$M_w$',
        'Tes': r'$T_{es}$',
        'Teh': r'$T_{eh}$',
        'ts': r'$t_s$',
        'th': r'$t_h$',
        'ta': r'$t_a$',
        'Surfactant_Mass': r'Mass$_{surf}$',
        'H2O/TEOS_molar': r'$H_2O$/TEOS',
        'SBET': r'$S_{BET}$',
        'DP': r'$D_P$',
        'VT': r'$V_T$'
    }
    
    renamed_edges = [(rename_map.get(u, u), rename_map.get(v, v)) for u, v in edges]
    
    # 5. Network Visualization
    plt.figure(figsize=(12, 10))
    G = nx.DiGraph()
    G.add_edges_from(renamed_edges)
    
    edge_labels = {}
    edge_colors = []
    edge_widths = []
    
    for u_raw, v_raw in edges:
        u_renamed = rename_map.get(u_raw, u_raw)
        v_renamed = rename_map.get(v_raw, v_raw)
        corr = df_dag[u_raw].corr(df_dag[v_raw])
        edge_labels[(u_renamed, v_renamed)] = f"{corr:.2f}"
        edge_colors.append('#d62728' if corr > 0 else '#1f77b4')
        edge_widths.append(1.0 + 3.0 * abs(corr))
        
    pos = nx.spring_layout(G, k=1.2, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_size=3500, node_color='#e0f3f8', edgecolors='#313695', linewidths=2)
    nx.draw_networkx_edges(G, pos, arrowsize=25, edge_color=edge_colors, width=edge_widths,
                           connectionstyle='arc3,rad=0.05', node_size=3500)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color='darkgreen', label_pos=0.5)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig67_causal_dag.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    build_causal_dag()
