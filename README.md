# Machine Learning Methods for Predicting March Madness Tournament Outcomes

This repository contains the code and analysis for a CS 536 final project on predicting NCAA March Madness outcomes for the men's and women's tournaments. The project treats bracket prediction as a matchup-by-matchup machine learning problem. Each game is converted into a Team 1 vs. Team 2 feature row, and models are trained to predict point differential and convert that prediction into a win probability.

## Project Goals

1. **Preprocess NCAA basketball data** into a modeling table where each row represents a matchup between two teams.
2. **Build predictive models** using seed information, regular season statistics, ranking systems, PageRank-style graph ratings, and XGBoost.
3. **Analyze second-chance brackets** beginning at the Sweet 16, where predictions are updated after the first two tournament rounds have already been played.

## Repository Structure

```text
.
├── data/                         # Kaggle March Machine Learning Mania CSV files
├── M\\\_preprocessing.ipynb          # Men's preprocessing and feature engineering
├── W\\\_preprocessing.ipynb          # Women's preprocessing and feature engineering
├── M\\\_R64\\\_analysis.ipynb           # Men's full 64-team tournament analysis
├── W\\\_R64\\\_analysis.ipynb           # Women's full 64-team tournament analysis
├── M\\\_R16\\\_analysis.ipynb           # Men's Sweet 16 / second-chance analysis
├── W\\\_R16\\\_analysis.ipynb           # Women's Sweet 16 / second-chance analysis
├── README.md                      # Project documentation
└── final\\\_writeup.pdf              # Final project report, if included
```

Some local notebook names may differ slightly, such as `M\\\_R16\\\_analysis(1).ipynb`, depending on how files were downloaded or exported.

## Data

The project uses the Kaggle March Machine Learning Mania dataset. The expected CSV files include regular season detailed results, NCAA tournament detailed results, seeds, teams, conferences, coaches, tournament slots, and ranking metrics.

Important files include:

### Men's Data

* `MRegularSeasonDetailedResults.csv`
* `MNCAATourneyDetailedResults.csv`
* `MNCAATourneySeeds.csv`
* `MTeams.csv`
* `MMasseyOrdinals.csv`
* `MTeamConferences.csv`
* `MTeamCoaches.csv`
* `MNCAATourneySlots.csv`
* `MNCAATourneySeedRoundSlots.csv`

### Women's Data

* `WRegularSeasonDetailedResults.csv`
* `WNCAATourneyDetailedResults.csv`
* `WNCAATourneySeeds.csv`
* `WTeams.csv`
* `WTeamConferences.csv`
* `WNCAATourneySlots.csv`

The notebooks expect these files to be located inside a folder named `data/`.

If the repository is hosted on GitHub, large CSV files may need to be tracked with Git LFS rather than committed normally.

## Environment Setup

The code was written in Python using Jupyter notebooks.

Install the main dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy xgboost tqdm networkx
```

If using Conda:

```bash
conda create -n march-madness-ml python=3.11
conda activate march-madness-ml
pip install pandas numpy matplotlib seaborn scikit-learn scipy xgboost tqdm networkx jupyter
```

Then launch Jupyter:

```bash
jupyter notebook
```

## Method Overview

### 1\. Matchup Preprocessing

The original NCAA results files store each game using winner and loser columns, such as `WTeamID`, `WScore`, `LTeamID`, and `LScore`. This format is not directly ideal for machine learning because the winning team is always listed first.

To avoid this issue, the preprocessing code creates two rows for each game:

* winner as Team 1, loser as Team 2
* loser as Team 1, winner as Team 2

The target variable is point differential:

```text
PointDiff = Team1Score - Team2Score
```

A positive point differential means Team 1 won, while a negative point differential means Team 1 lost.

### 2\. Regular Season Feature Aggregation

For each team and season, the notebooks compute average regular season statistics, including:

* average score
* average field goals made and attempted
* average three-pointers made and attempted
* average free throws made and attempted
* average rebounds
* average assists
* average turnovers
* average steals
* average blocks
* average fouls
* average point differential
* opponent averages

These season-level averages are merged into each tournament matchup so that every tournament game contains both teams' pre-tournament statistical profiles.

### 3\. Seed and Ranking Features

Tournament seed is converted from strings such as `W01` or `X12` into numeric seed values. The notebooks create:

```text
T1\\\_seed
T2\\\_seed
seed\\\_Diff = T2\\\_seed - T1\\\_seed
```

Ranking systems from Massey Ordinals are also considered, including metrics such as POM, MOR, NET, WAB, and others when available.

### 4\. PageRank Team Strength

The project adapts PageRank to college basketball by treating teams as nodes in a directed graph.

For each regular season:

* each team is a node
* a loss creates a directed edge from the losing team to the winning team
* edge weights are based on average margin of loss
* each team's outgoing edge weights are normalized to sum to 1

The PageRank equation is:

```text
v = dMv + (1 - d) / N
```

where:

* `v` is the PageRank vector
* `M` is the directed transition matrix
* `d` is the damping factor
* `N` is the number of teams

The resulting `PR\\\_Power` values are merged into tournament matchups as:

```text
T1\\\_PageRank
T2\\\_PageRank
PageRank\\\_Diff
```

### 5\. XGBoost Regression

The main prediction model is XGBoost regression. Instead of directly predicting win or loss, the model predicts point differential. This gives more information than a binary label because a predicted win by 12 points is more confident than a predicted win by 1 point.

The notebooks use XGBoost with parameters similar to:

```python
param = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "eta": 0.01,
    "subsample": 0.6,
    "colsample\\\_bynode": 0.8,
    "num\\\_parallel\\\_tree": 2,
    "min\\\_child\\\_weight": 4,
    "max\\\_depth": 4,
    "tree\\\_method": "hist",
    "grow\\\_policy": "lossguide",
    "max\\\_bin": 32
}
```

### 6\. Win Probability Calibration

Predicted point differentials are converted into win probabilities using a spline fit. The idea is to estimate how often teams actually win when the model predicts a certain margin.

For example, if teams predicted to win by around 10 points historically win about 80% of the time, then a predicted point differential of 10 maps to a win probability of about 0.80.

Probabilities are clipped between 0.01 and 0.99 to avoid extreme values.

## Validation and Metrics

The notebooks use season-based holdout validation. For each season, the model trains on all other seasons and tests on the held-out season. This better matches the real task of predicting a future tournament from past tournaments.

The main evaluation metrics are:

### Mean Absolute Error

Measures how many points off the point differential prediction is on average.

```text
MAE = mean(|actual point differential - predicted point differential|)
```

### Brier Score

Measures the quality of predicted win probabilities.

```text
Brier = mean((predicted probability - actual outcome)^2)
```

Lower Brier scores are better.

## Notebook Guide

### `M\\\_preprocessing.ipynb`

Builds the men's matchup dataset. Includes data loading, winner/loser transformation, regular season aggregation, seed merging, PageRank feature creation, and exploratory correlation plots.

### `W\\\_preprocessing.ipynb`

Builds the women's matchup dataset using a similar process, beginning from the women's data files and focusing on seasons where detailed data is available.

### `M\\\_R64\\\_analysis.ipynb`

Runs the men's full-tournament analysis. Tests different feature sets, trains XGBoost models with season-based validation, calculates MAE and Brier score, and analyzes feature importance.

### `W\\\_R64\\\_analysis.ipynb`

Runs the women's full-tournament analysis. Uses a similar modeling process to the men's analysis but trains and evaluates separately because the men's and women's tournaments have different upset patterns and feature signals.

### `M\\\_R16\\\_analysis.ipynb`

Runs the men's Sweet 16 / second-chance bracket analysis. The first two rounds are treated as already observed, and the model predicts outcomes from the Sweet 16 onward.

### `W\\\_R16\\\_analysis.ipynb`

Runs the women's Sweet 16 / second-chance bracket analysis using the women's tournament data.

## Key Findings

* Ranking and seed-based features are consistently important predictors.
* PageRank provides a useful graph-based team strength signal and often appears as an important model feature.
* Raw offensive statistics generally performed better than raw defensive-only statistics, although this may partly reflect that offensive box score stats are more directly tied to point differential.
* Women's tournament predictions produced lower Brier scores but higher MAE values, suggesting different prediction behavior than the men's tournament.
* Sweet 16 / second-chance predictions are harder because the remaining teams are closer in strength.
* Moderate-size feature sets often performed nearly as well as very large feature sets, suggesting that many features encode overlapping information.

## Assumptions and Limitations

* Regular season averages are assumed to be useful indicators of tournament performance.
* The model does not fully account for injuries, roster changes, late-season form, coaching adjustments, or matchup-specific strategy.
* PageRank depends on the chosen edge weighting method and damping factor.
* Many ranking systems are correlated with each other, which can make feature importance harder to interpret.
* NCAA tournament sample sizes are small, especially for later rounds.
* Bracket games are modeled mostly as independent matchups, even though fatigue, travel, injuries, and previous tournament performance may affect later rounds.

## Suggested Run Order

1. Place all Kaggle CSV files inside `data/`.
2. Run `M\\\_preprocessing.ipynb` and/or `W\\\_preprocessing.ipynb` to inspect preprocessing and features.
3. Run `M\\\_R64\\\_analysis.ipynb` for men's full tournament experiments.
4. Run `W\\\_R64\\\_analysis.ipynb` for women's full tournament experiments.
5. Run `M\\\_R16\\\_analysis.ipynb` and `W\\\_R16\\\_analysis.ipynb` for second-chance bracket experiments.

## Authors

* Krishaan Chaudhary
* Eashan Patel

CS 536 Final Project

