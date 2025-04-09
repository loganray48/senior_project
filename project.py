# car_price_stacking.py (Updated & Optimized)
#%%
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostClassifier
from autogluon.tabular import TabularPredictor

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train = train.drop(columns=["id"])

# Separate target
y = train["price"]
X = train.drop(columns=["price"])


#%%
# SECTION 1: Feature Engineering and Cleaning

# Fill missing values
X['clean_title'] = X['clean_title'].fillna("Unknown")
X['fuel_type'] = X['fuel_type'].fillna("Unknown")
X['accident'] = X['accident'].fillna("None reported")
X['transmission'] = X['transmission'].fillna("Unknown")

# Simplify transmission
X['transmission_simplified'] = X['transmission'].replace({
    'Transmission w/Dual Shift Mode': 'A/T',
    '10-Speed Automatic': 'A/T',
    '6-Speed M/T': 'M/T',
    '7-Speed A/T': 'A/T',
    'Automatic': 'A/T',
    'Manual': 'M/T'
})

# Car age and mileage per year
X['car_age'] = 2025 - X['model_year']
X['mileage_per_year'] = X['milage'] / X['car_age'].replace(0, 1)

# Luxury brand flag
luxury_brands = ["BMW", "Mercedes-Benz", "Audi", "Porsche", "Lexus", "Genesis", "Jaguar"]
X['is_luxury'] = X['brand'].isin(luxury_brands).astype(int)

# Simplify model
X['model_simplified'] = X['model'].str.split().str[0]

# Feature crosses
X['int_ext_col'] = X['int_col'] + '_' + X['ext_col']
X['brand_model'] = X['brand'] + '_' + X['model_simplified']
X['brand_int_col'] = X['brand'] + '_' + X['int_col']
X['brand_ext_col'] = X['brand'] + '_' + X['ext_col']
X['brand_mileage'] = X['brand'] + '_' + pd.qcut(X['milage'], q=5, duplicates='drop').astype(str)

# Parse engine for horsepower, displacement, cylinders
def extract_engine_features(df):
    df['horsepower'] = df['engine'].str.extract(r'(\d+\.?\d*)HP').astype(float)
    df['displacement'] = df['engine'].str.extract(r'(\d+\.\d+)L').astype(float)
    df['cylinders'] = df['engine'].str.extract(r'(\d+) Cylinder').astype(float)
    return df

X = extract_engine_features(X)

# Rare category flagging
cat_features = X.select_dtypes(include="object").columns.tolist()
def label_encode_and_group_rare(X, cat_features, threshold=0.01):
    for col in cat_features:
        freq = X[col].value_counts(normalize=True)
        rare = freq[freq < threshold].index
        X[col] = X[col].replace(rare, 'rare')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    return X

X_label_encoded = label_encode_and_group_rare(X.copy(), cat_features)


#%%
# SECTION 2: Outlier Classification Feature
def bin_price(data):
    df = data.copy()
    Q1 = np.percentile(df['price'], 25)
    Q3 = np.percentile(df['price'], 75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    df['price_bin'] = (df['price'] < upper_bound).astype(int)
    return df

train_bin = bin_price(train)

X[cat_features] = X[cat_features].fillna("missing")
clf = CatBoostClassifier(
    iterations=200,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    verbose=0
)
clf.fit(X, train_bin['price_bin'])
outlier_pred = clf.predict_proba(X)[:, 1]
X_label_encoded['outlier_prob'] = outlier_pred




#%%
# SECTION 3: Median Target Encoding (Leak-Free)
def target_encode_median_leakfree(X, y, colname, folds):
    X_encoded = pd.Series(index=X.index, dtype=float)
    for train_idx, val_idx in folds.split(X):
        median_map = y.iloc[train_idx].groupby(X.iloc[train_idx][colname]).median()
        X_encoded.iloc[val_idx] = X.iloc[val_idx][colname].map(median_map)
    return X_encoded

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for col in cat_features:
    X_label_encoded[f"{col}_med_te"] = target_encode_median_leakfree(X, y, col, kf)



#%%
# SECTION 4: Base Models + OOF Predictions
def get_oof_predictions(X, y, model, model_name, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    for train_idx, val_idx in kf.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        oof_preds[val_idx] = preds
    return oof_preds

xgb = XGBRegressor(tree_method='hist', device='cuda', learning_rate=0.05, n_estimators=300)
lgbm = LGBMRegressor(device='gpu', learning_rate=0.05, n_estimators=300)
hgb = HistGradientBoostingRegressor(max_iter=300)

X_base = X_label_encoded.copy()
X_base['xgb_oof'] = get_oof_predictions(X_base, y, xgb, "xgb")
X_base['lgbm_oof'] = get_oof_predictions(X_base, y, lgbm, "lgbm")
X_base['hgb_oof'] = get_oof_predictions(X_base.fillna(-999), y, hgb, "hgb")



#%%
# SECTION 5: AutoGluon FastAI Model
X_autogluon = X.copy()
X_autogluon['price'] = y
predictor = TabularPredictor(label='price', eval_metric='rmse').fit(
    X_autogluon,
    time_limit=300,
    presets="best_quality",
    included_model_types=["FASTAI"]
)
fastai_preds = predictor.predict(X)
X_base['fastai_pred'] = fastai_preds.values

# SECTION 6: Final Stacking with Ridge
stack = Ridge(alpha=1.0)
stack.fit(X_base[['xgb_oof', 'lgbm_oof', 'hgb_oof', 'fastai_pred']], y)
y_pred_final = stack.predict(X_base[['xgb_oof', 'lgbm_oof', 'hgb_oof', 'fastai_pred']])
print(f"Final Ensemble RMSE: {mean_squared_error(y, y_pred_final, squared=False):.4f}")

# %%
