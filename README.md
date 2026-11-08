# Used Cars: Price Analysis & Prediction

Exploratory analysis and price prediction on 8,128 used-car listings (brand, mileage, fuel type, number of owners, selling price).

## What this project does
- **Data cleaning with justified decisions**: removes 20% duplicate rows, drops the 5 "Test Drive Car" listings and implausible mileage, and keeps luxury cars (modelled on a log-price scale instead of clipping them).
- **EDA**: price distributions (raw vs log), price by brand / fuel / owner count.
- **Statistics**: Spearman correlation between mileage and price per fuel type, with sample sizes and 95% confidence intervals.
- **Modelling**: Ridge baseline vs Random Forest vs Gradient Boosting with 5-fold CV, then a held-out test set and permutation feature importance.

## Results
| Model | CV R² (log price) |
|---|---|
| Ridge (baseline) | 0.53 |
| Random Forest | 0.49 |
| Gradient Boosting | **0.56** |

Test set: R² = 0.56, MAE about 187k, median absolute error about 28%.
Mileage is negatively correlated with price for every fuel type (Spearman -0.59 to -0.42), but it is only one factor among several. Fuel type and brand matter most in the model.

## Limitations
The dataset has no car age/year, engine size or transmission. Adding them would likely improve the model the most. LPG (38 cars) and CNG (54) results are based on small samples.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook cars_price_analysis.ipynb
```
Data: `data/cars_dataset.csv`
