# Machine Learning Madness

A machine learning project that predicts NCAA March Madness matchups using regular-season statistics, ranking systems, PageRank-style team strength, and XGBoost.

## Project Description

This project builds a matchup-based prediction pipeline for the men's and women's NCAA basketball tournaments. It preprocesses Kaggle March Machine Learning Mania data into team-vs-team feature tables, engineers statistics such as seed difference, regular-season averages, ranking differentials, and PageRank power scores, then trains XGBoost models to predict tournament point differential and win probability.

The motivation is to study whether machine learning can improve March Madness bracket prediction beyond simple seed-based picks, while also comparing full-bracket predictions with second-chance Sweet 16 predictions.

## Installation Instructions

Clone the repository:

```bash
git clone https://github.com/KriChau95/machine-learning-madness.git
cd machine-learning-madness
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost tqdm jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

## Quick Start

1. Download the Kaggle March Machine Learning Mania dataset. (https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data)
2. Place the CSV files into a local `data/` folder in the project directory.
3. Open the notebooks in Jupyter.
4. Run the model analysis notebooks:

```text
M_R64_analysis.ipynb
W_R64_analysis.ipynb
M_R16_analysis.ipynb
W_R16_analysis.ipynb
```

The Round of 64 notebooks evaluate full tournament prediction, while the Round of 16 notebooks evaluate second-chance bracket prediction.

## Requirements / Dependencies

- Python 3.10+
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
- xgboost
- tqdm
- jupyter

