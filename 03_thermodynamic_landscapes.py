import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-paper')

OUTPUT_DIR = "."
DATA_FILE = "../../ml_pipeline/processed_data/universal_synthesis_data.csv"

def generate_thermodynamic_plots():
    """
    Trains a Universal Random Forest model to generate 3D thermodynamic 
    response surfaces and pseudo-Arrhenius kinetic plots.
    """
    df = pd.read_csv(DATA_FILE)
    
    # Isolate targets and features
    X = df.drop(columns=['SBET', 'DP', 'VT', 'Acidity_Molar'])
    features = X.columns.tolist()
    
    # 1. Train Generic Random Forest Regressors
    models = {}
    for target in ['SBET', 'DP', 'VT']:
        rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
        rf.fit(X, df[target])
        models[target] = rf
        
    medians = X.median().to_dict()
    
    for target in ['SBET', 'DP', 'VT']:
        
        # ---------------------------------------------------------
        # Fig 69: 3D Landscape for CTAB (MCM-41)
        # ---------------------------------------------------------
        temps_ctab = np.linspace(20, 100, 30)
        times_ctab = np.linspace(1, 48, 30)
        T_grid_c, t_grid_c = np.meshgrid(temps_ctab, times_ctab)
        Z_ctab = np.zeros_like(T_grid_c)
        
        for i in range(T_grid_c.shape[0]):
            for j in range(T_grid_c.shape[1]):
                recipe = medians.copy()
                # Lock in CTAB topology and pH
                recipe['Surf_TPSA'] = 0.0
                recipe['Estimated_pH'] = 9.51
                # Vary Synthesis variables
                recipe['Tes'] = T_grid_c[i, j]
                recipe['ts'] = t_grid_c[i, j]
                # Nullify Hydrothermal phase
                recipe['Teh'] = 0
                recipe['th'] = 0
                
                df_pred = pd.DataFrame([recipe])[features]
                Z_ctab[i, j] = models[target].predict(df_pred)[0]
                
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(T_grid_c, t_grid_c, Z_ctab, cmap='viridis', edgecolor='none', alpha=0.8)
        ax.set_xlabel('Reaction Temperature (°C)')
        ax.set_ylabel('Reaction Time (h)')
        ax.set_zlabel(f'Predicted {target}')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'Fig69_landscape_CTAB_{target}.png'), dpi=300)
        plt.close()
        
        # ---------------------------------------------------------
        # Fig 70: 3D Landscape for Pluronic (SBA-15)
        # ---------------------------------------------------------
        temps_plur = np.linspace(40, 150, 30)
        times_plur = np.linspace(1, 48, 30)
        T_grid_p, t_grid_p = np.meshgrid(temps_plur, times_plur)
        Z_plur = np.zeros_like(T_grid_p)
        
        for i in range(T_grid_p.shape[0]):
            for j in range(T_grid_p.shape[1]):
                recipe = medians.copy()
                # Lock in Pluronic topology and pH
                recipe['Surf_TPSA'] = 2477.2
                recipe['Estimated_pH'] = 1.16
                # Vary Hydrothermal variables
                recipe['Teh'] = T_grid_p[i, j]
                recipe['th'] = t_grid_p[i, j]
                # Lock early synthesis phase
                recipe['Tes'] = 40
                recipe['ts'] = 24
                
                df_pred = pd.DataFrame([recipe])[features]
                Z_plur[i, j] = models[target].predict(df_pred)[0]
                
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(T_grid_p, t_grid_p, Z_plur, cmap='magma', edgecolor='none', alpha=0.8)
        ax.set_xlabel('Reaction Temperature (°C)')
        ax.set_ylabel('Reaction Time (h)')
        ax.set_zlabel(f'Predicted {target}')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'Fig70_landscape_Pluronic_{target}.png'), dpi=300)
        plt.close()
        
        # ---------------------------------------------------------
        # Fig 71: Universal Arrhenius Kinetics
        # ---------------------------------------------------------
        temps_arrhenius = np.linspace(30, 150, 50)
        inverse_T = 1000 / (temps_arrhenius + 273.15)
        
        arr_ctab = []
        arr_plur = []
        
        for T in temps_arrhenius:
            rec_c = medians.copy()
            rec_c['Surf_TPSA'] = 0.0
            rec_c['Estimated_pH'] = 9.51
            rec_c['Tes'] = T
            rec_c['ts'] = 24
            arr_ctab.append(models[target].predict(pd.DataFrame([rec_c])[features])[0])
            
            rec_p = medians.copy()
            rec_p['Surf_TPSA'] = 2477.2
            rec_p['Estimated_pH'] = 1.16
            rec_p['Teh'] = T
            rec_p['th'] = 24
            arr_plur.append(models[target].predict(pd.DataFrame([rec_p])[features])[0])
            
        plt.figure(figsize=(7, 5))
        plt.plot(inverse_T, np.log(arr_ctab), 'o-', color='teal', label='CTAB')
        plt.plot(inverse_T, np.log(arr_plur), 's-', color='coral', label='Pluronic F127')
        plt.xlabel('1000 / T ($K^{-1}$)')
        plt.ylabel(f'ln({target})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'Fig71_arrhenius_{target}.png'), dpi=300)
        plt.close()

if __name__ == '__main__':
    generate_thermodynamic_plots()
