import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import pi
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-paper')

OUTPUT_DIR = "."
DATA_FILE = "../../ml_pipeline/processed_data/universal_synthesis_data.csv"

# Global colors matching the manuscript scheme
COLORS = {
    'CTAB': '#1f77b4',           # Blue
    'Pluronic F127': '#d62728'   # Red
}

def compute_green_metrics(df):
    """
    Calculates E-factor, Process Mass Intensity (PMI), and Energy Proxy 
    based on standard green chemistry equations for mesoporous silica.
    """
    M_SiO2 = 60.08
    M_TEOS = 208.33
    M_H2O = 18.015
    
    # Mass of inputs (assuming 1 mol TEOS base)
    mass_teos = M_TEOS
    mass_h2o = df['H2O/TEOS_molar'] * M_H2O
    mass_surf = df['Surfactant_Mass']
    
    # 1. E-factor: (Total Mass In - Mass Product) / Mass Product
    # Assuming ~100% yield for TEOS -> SiO2 condensation
    total_mass_in = mass_teos + mass_h2o + mass_surf
    mass_product = M_SiO2
    df['E_factor'] = (total_mass_in - mass_product) / mass_product
    
    # 2. PMI: Total Mass In / Mass Product
    df['PMI'] = total_mass_in / mass_product
    
    # 3. Energy Proxy (Heating footprint: Temperature * Time)
    df['Energy_Proxy'] = (df['Tes'] * df['ts']) + (df['Teh'] * df['th']) + (df['ta'] * 25)
    
    return df

def generate_green_radar():
    """
    Simulates recipes, extracts optimal green recipes for both ionic and non-ionic templates,
    and plots a comparative radar chart.
    """
    df_raw = pd.read_csv(DATA_FILE)
    
    # 1. Train Universal RF on actual data
    X = df_raw.drop(columns=['SBET', 'DP', 'VT', 'Acidity_Molar'])
    features = X.columns.tolist()
    
    models = {}
    for target in ['SBET', 'DP', 'VT']:
        rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
        rf.fit(X, df_raw[target])
        models[target] = rf
        
    # 2. Simulate a grid of possible recipes to find the optimal ones
    sim_data = []
    medians = X.median().to_dict()
    
    # Generate 50 CTAB recipes
    for _ in range(50):
        rec = medians.copy()
        rec['Surf_TPSA'] = 0.0
        rec['Estimated_pH'] = np.random.uniform(9.0, 11.0)
        rec['Surfactant_Mass'] = np.random.uniform(0.5, 3.0)
        rec['Tes'] = np.random.uniform(20, 80)
        rec['ts'] = np.random.uniform(2, 48)
        rec['Teh'] = 0
        rec['th'] = 0
        rec['H2O/TEOS_molar'] = np.random.uniform(50, 150)
        rec['Surfactant'] = 'CTAB'
        sim_data.append(rec)
        
    # Generate 50 Pluronic recipes
    for _ in range(50):
        rec = medians.copy()
        rec['Surf_TPSA'] = 2477.2
        rec['Estimated_pH'] = np.random.uniform(1.0, 2.0)
        rec['Surfactant_Mass'] = np.random.uniform(1.0, 5.0)
        rec['Tes'] = 40
        rec['ts'] = 24
        rec['Teh'] = np.random.uniform(60, 150)
        rec['th'] = np.random.uniform(12, 48)
        rec['H2O/TEOS_molar'] = np.random.uniform(100, 250)
        rec['Surfactant'] = 'Pluronic F127'
        sim_data.append(rec)
        
    df_sim = pd.DataFrame(sim_data)
    X_sim = df_sim[features]
    
    for target in ['SBET', 'DP', 'VT']:
        df_sim[f'Pred_{target}'] = models[target].predict(X_sim)
        
    # Calculate green metrics
    df_sim = compute_green_metrics(df_sim)
    
    # Calculate Green Score to find the 'Optimal' recipe
    def ni(s): return 1.0 - (s - s.min()) / (s.max() - s.min() + 1e-9)
    def nd(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)
    
    # Surfactant Efficiency
    df_sim['SE_SBET'] = df_sim['Pred_SBET'] / df_sim['Surfactant_Mass']
    
    df_sim['Green_Score'] = (
        0.30 * ni(df_sim['E_factor']) +
        0.25 * ni(df_sim['Energy_Proxy']) +
        0.25 * nd(df_sim['SE_SBET']) +
        0.20 * ni(df_sim['PMI'])
    ) * 10.0
    
    best_ctab = df_sim[df_sim['Surfactant'] == 'CTAB'].nlargest(1, 'Green_Score').iloc[0]
    best_plur = df_sim[df_sim['Surfactant'] == 'Pluronic F127'].nlargest(1, 'Green_Score').iloc[0]
    
    # 3. Plot Radar Chart (Fig 72)
    categories = ['S$_{BET}$', 'D$_P$', 'V$_T$', '1 / E-factor\n(Low Waste)', '1 / PMI\n(Mass Eff.)', '1 / Energy\n(Low Cost)']
    N = len(categories)
    
    def norm_val(val, col, invert=False):
        mn, mx = df_sim[col].min(), df_sim[col].max()
        if invert:
            v = 1.0 - (val - mn) / (mx - mn + 1e-9)
        else:
            v = (val - mn) / (mx - mn + 1e-9)
        return 0.1 + 0.9 * v
        
    vals_ctab = [
        norm_val(best_ctab['Pred_SBET'], 'Pred_SBET'),
        norm_val(best_ctab['Pred_DP'], 'Pred_DP'),
        norm_val(best_ctab['Pred_VT'], 'Pred_VT'),
        norm_val(best_ctab['E_factor'], 'E_factor', invert=True),
        norm_val(best_ctab['PMI'], 'PMI', invert=True),
        norm_val(best_ctab['Energy_Proxy'], 'Energy_Proxy', invert=True)
    ]
    vals_plur = [
        norm_val(best_plur['Pred_SBET'], 'Pred_SBET'),
        norm_val(best_plur['Pred_DP'], 'Pred_DP'),
        norm_val(best_plur['Pred_VT'], 'Pred_VT'),
        norm_val(best_plur['E_factor'], 'E_factor', invert=True),
        norm_val(best_plur['PMI'], 'PMI', invert=True),
        norm_val(best_plur['Energy_Proxy'], 'Energy_Proxy', invert=True)
    ]
    
    vals_ctab += vals_ctab[:1]
    vals_plur += vals_plur[:1]
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, size=11, fontweight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["", "", "", ""], color="grey", size=7)
    plt.ylim(0, 1.05)
    
    ax.plot(angles, vals_ctab, linewidth=2, linestyle='solid', color=COLORS['CTAB'], label='Optimal CTAB')
    ax.fill(angles, vals_ctab, color=COLORS['CTAB'], alpha=0.25)
    
    ax.plot(angles, vals_plur, linewidth=2, linestyle='solid', color=COLORS['Pluronic F127'], label='Optimal Pluronic F127')
    ax.fill(angles, vals_plur, color=COLORS['Pluronic F127'], alpha=0.25)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig_green_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    generate_green_radar()
