# WNBA Draft Waiver Prediction

This project uses machine learning to predict whether a WNBA draft pick will be waived before playing for their team, based on team, position, and draft round. It focuses on draftees from 2020 to 2024 and combines publicly available data with manually curated waiver information.

## Tools & Libraries
- **Python (Google Colab)**
- **pandas** for data manipulation
- **scikit-learn** for machine learning
- **matplotlib / seaborn** for visualization
- **LabelEncoder** for encoding categorical variables
- **RandomForestClassifier** for classification

## Data Sources
- `enriched_wnba_draftees.csv`: Draft data from 1997–2024 (cleaned version of a Kaggle dataset)
- `WNBADraft_20_24.csv`: Custom dataset of players waived between 2020–2024 before playing a game

## Model Overview
- Features: draft round, team (encoded), and position (encoded)
- Target: whether a player was waived
- Classifier: RandomForestClassifier with `class_weight='balanced'`
- Evaluation: Model trained on an 80/20 train-test split

## Sample Prediction

```python
predict_waiver("Indiana Fever", "guard", 2)
