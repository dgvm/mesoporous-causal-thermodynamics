# Machine Learning Pipeline for Mesoporous Silica Synthesis

This repository contains the Python scripts required to reproduce the machine learning models, causal networks, and thermodynamic landscapes presented in the manuscript. The code applies multi-output ensemble algorithms and Bayesian structure learning to extract deterministic physicochemical rules from aggregated synthesis data.

## Prerequisites

Ensure the following Python dependencies are installed prior to execution:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn networkx pgmpy
```

## Data Requirements

The scripts require the preprocessed dataset `universal_synthesis_data.csv` to be located in a valid relative path (configured within each script). This dataset must contain the unified operational variables (e.g., $T_{es}$, $t_s$, $T_{eh}$, $t_h$, Surfactant Mass) and the continuous chemical descriptors ($M_w$, $\log P$, TPSA, $\widetilde{pH}$) alongside the target textural properties ($S_{BET}$, $D_P$, $V_T$).

## Code Structure and Usage

The repository is organized into four independent modules. They can be executed sequentially or individually depending on the required analysis.

### 1. Causal Structure Learning
**Script:** `01_causal_dag.py`
* **Function:** Implements a Hill-Climbing algorithm with a Bayesian Information Criterion (BIC) scoring method to infer a Directed Acyclic Graph (DAG).
* **Output:** Generates the causal hierarchy network, mapping the physical sequence of events during synthesis.

### 2. Non-Linear Interaction Network
**Script:** `02_interaction_network.py`
* **Function:** Trains independent Random Forest Regressors to extract Gini feature importances, constructing a weighted bipartite graph of non-linear interactions.
* **Output:** Generates the operational influence network, visualizing the predominant drivers for each textural property.

### 3. Thermodynamic Landscapes
**Script:** `03_thermodynamic_landscapes.py`
* **Function:** Deploys a universal ensemble simulator to evaluate a dense predictive grid, holding surfactant topologies constant while mapping reaction kinetics.
* **Output:** Generates 3D response surfaces for CTAB and Pluronic F127, along with pseudo-Arrhenius kinetic extrapolations.

### 4. Green Chemistry Optimization
**Script:** `04_green_chemistry_metrics.py`
* **Function:** Computes standardized sustainability metrics (E-factor, Process Mass Intensity, Energy Proxy) and identifies the Pareto-optimal recipes via a multi-objective scoring function.
* **Output:** Generates a comparative radar chart mapping the environmental and performance footprint of the optimal synthesis routes.

## Execution

To generate the figures, execute any script directly via the command line. Outputs are saved automatically in the working directory as high-resolution PNG files.

```bash
python 01_causal_dag.py
```
