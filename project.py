
#%%
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import cupy as cp  # Ensure CuPy is installed for GPU acceleration

# Load Data
train = pd.read_csv("train.csv")
X = train.drop(columns=["price", "id"])
y = train["price"]

# Identify Numerical & Categorical Features
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocessing Pipeline (keep data in Pandas here)
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)  # Ensure dense format
])

# Train/Test Split (still in Pandas)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# **Apply Preprocessing First (Convert Data to Numeric Format)**
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)

# **Now Move to GPU (Convert to CuPy Arrays)**
X_train = cp.asarray(X_train)  # Convert only after preprocessing
X_val = cp.asarray(X_val)
y_train = cp.asarray(y_train)
y_val = cp.asarray(y_val)

# **Use XGBoost with GPU**
model = XGBRegressor(
    n_estimators=200,         # Increase for better accuracy
    learning_rate=0.05,       # Adjust for tuning
    tree_method="hist",       # Optimized tree method
    device="cuda"             # GPU acceleration
)

# Train Model
model.fit(X_train, y_train)

# Evaluate Model
y_val_pred = model.predict(X_val)
mse = mean_squared_error(cp.asnumpy(y_val), cp.asnumpy(y_val_pred))  # Convert back to CPU for metrics
r2 = r2_score(cp.asnumpy(y_val), cp.asnumpy(y_val_pred))

print(f"MSE: {mse}, R²: {r2}")


# %%
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import cupy as cp  # CuPy for GPU acceleration

# Load Data
train = pd.read_csv("train.csv")
X = train.drop(columns=["price", "id"])
y = train["price"]

# Identify Numerical & Categorical Features
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
])

# Train/Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# **Apply Preprocessing First**
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)

# **Convert to CuPy for GPU Speedup**
X_train = cp.asarray(X_train)
X_val = cp.asarray(X_val)
y_train = cp.asarray(y_train)
y_val = cp.asarray(y_val)

# **Convert Back to NumPy Before LightGBM**
X_train = cp.asnumpy(X_train)
X_val = cp.asnumpy(X_val)
y_train = cp.asnumpy(y_train)
y_val = cp.asnumpy(y_val)

# **Use LightGBM with GPU**
model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=-1,  # Let model decide
    num_leaves=100,
    subsample=0.8,
    colsample_bytree=0.8,
    device="gpu"  # Run on GPU
)

# Train Model
model.fit(X_train, y_train)

# Evaluate Model
y_val_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print(f"MSE: {mse}, R²: {r2}")

# %%
