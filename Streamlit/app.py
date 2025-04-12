import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

st.set_page_config(page_title="Car Price Regression Dashboard", layout="wide")
# Load Data
@st.cache_data
def load_data():
    train = pd.read_csv("../train.csv")
    train['clean_title'] = train['clean_title'].fillna("Unknown")
    train['fuel_type'] = train['fuel_type'].fillna("Unknown")
    train['accident'] = train['accident'].fillna("None reported")
    train['car_age'] = 2025 - train['model_year']
    train['mileage_per_year'] = train['milage'] / train['car_age'].replace(0, 1)
    train['transmission'] = train['transmission'].fillna("Unknown")
    train['transmission_simplified'] = train['transmission'].replace({
        'Transmission w/Dual Shift Mode': 'A/T',
        '10-Speed Automatic': 'A/T',
        '6-Speed M/T': 'M/T',
        '7-Speed A/T': 'A/T',
        'Automatic': 'A/T',
        'Manual': 'M/T'
    })
    return train

train = load_data()

# App Layout
st.title("Car Price Regression | Senior Project")

# Tabs
tab1, tab2, tab3 = st.tabs(["Project Overview", "Data Validation", "ML Results"])

# Tab 1: Project Overview
with tab1:
    st.header("Project Introduction")
    st.markdown("""
    This project is part of my senior project where I participated in the Kaggle 2024 Tabular Playground Series
    to predict used car prices. The dataset includes synthetically generated records with attributes such as brand,
    mileage, fuel type, engine configuration, and more. The objective is to build a regression model that predicts the 
    price of a used car and explore how various features affect pricing. Multiple machine learning models and ensemble 
    techniques were tested, along with extensive feature engineering.
    """)

    st.markdown("## Feature Engineering & Data Augmentation")

    # Subsection 1: New Columns
    st.markdown("### New Columns Created")
    st.markdown("""
    To provide more predictive power to the model, several new features were derived from the original dataset:

    - **car_age** – Calculated as `2025 - model_year` to measure how old the vehicle is.
    - **mileage_per_year** – Estimated wear by dividing total mileage by car age (handles zero age safely).
    - **is_luxury** – Boolean column indicating whether the car brand is considered a luxury make (e.g., BMW, Audi, Lexus).
    - **transmission_simplified** – Collapsed multiple complex labels into two main types: `"A/T"` and `"M/T"`.
    - **Feature Crosses** – Combined features such as:
        - `brand_model`
        - `brand_int_col`
        - `brand_ext_col`
        - `brand_mileage` (mileage bucketed per brand)

    These features helped the models capture more domain-specific relationships between vehicle type and price.
    """)

    st.markdown("#### Preview of New Columns")
    st.dataframe(train.head(5))

    # Subsection 2: Other Augmentations
    st.markdown("### Additional Data Augmentations")
    st.markdown("""
    Alongside feature creation, we also performed several enhancements to clean and prepare the dataset:

    - **Missing Value Handling** – Imputed missing fields with meaningful defaults like `"Unknown"` or `"None reported"`.
    - **Outlier Flagging** – Used IQR to identify high-price outliers and trained a CatBoostClassifier to detect them.
    - **Target Encoding** – Median-encoded categorical columns using leak-free 5-fold cross-validation.
    - **OOF Model Features** – Generated out-of-fold (OOF) predictions from base models (XGBoost, LightGBM, SVR) to use in stacking.
    - **AutoML Predictions** – Injected predictions from AutoGluon's FastAI model as additional meta-features.
    - **Category Simplification** – Cleaned noisy or rare categories such as `"_"` and `"not supported"`.

    Together, these techniques significantly improved signal quality and reduced noise, helping the final ensemble achieve strong performance.
    """)


# Tab 2: Data Validation / EDA
with tab2:
    st.header("Exploratory Data Analysis")

#Figure 1
    st.subheader("1. Price Distribution")
    filtered = train[train["price"] <= 100000]
    fig1 = px.histogram(
        filtered,
        x="price",
        nbins=40,
        title="Distribution of Car Prices (Under $100K)",
    )
    fig1.update_xaxes(
        tick0=0,
        dtick=10000,
        tickformat="$,",
        title="Price ($)"
    )
    fig1.update_layout(
        yaxis_title="Count",
        xaxis=dict(range=[0, 100000])
    )
    st.plotly_chart(fig1, use_container_width=True)

#Figure 2
    st.subheader("2. Price by Car Age")
    filtered = train[(train["price"] <= 300000) & (train["car_age"] <= 33)]

    fig2 = px.box(
        filtered,
        x="car_age",
        y="price",
        points=False,
        title="Car Price by Car Age (Under $300K)"
    )
    fig2.update_yaxes(
        range=[0, 300000],
        tickformat="$,",
        title="Price ($)"
    )
    fig2.update_xaxes(title="Car Age (Years)")
    st.plotly_chart(fig2, use_container_width=True)

#Figure 3
    st.subheader("3. Average Price by Top 10 Brands")
    top_brands = train['brand'].value_counts().head(10).index
    brand_avg_price = train[train['brand'].isin(top_brands)].groupby('brand')['price'].mean().sort_values(ascending=False).reset_index()
    fig3 = px.bar(brand_avg_price, x='brand', y='price', title='Average Price by Top 10 Brands')
    st.plotly_chart(fig3, use_container_width=True)

#Figure 4
    st.subheader("4. Price by Fuel Type")
    # Filter prices under 100K
    filtered = train[train["price"] <= 100000].copy()
    # Clean and filter valid fuel types
    valid_fuels = [
        "Gasoline", "E85 Flex Fuel", "Hybrid", "Diesel", "Plug-In Hybrid", "Unknown", "not supported"
    ]
    filtered = filtered[filtered["fuel_type"].isin(valid_fuels)]
    filtered["fuel_type"] = filtered["fuel_type"].replace({
        "not supported": "Other",
        "Unknown": "Other"
    })
    fig4 = px.box(
        filtered,
        x="fuel_type",
        y="price",
        points=False,
        title="Price by Fuel Type (Under $100K)"
    )
    fig4.update_yaxes(
        range=[0, 100000],
        tickformat="$,",
        title="Price ($)"
    )
    fig4.update_xaxes(title="Fuel Type")
    st.plotly_chart(fig4, use_container_width=True)



#Figure 5
    st.subheader("5. Transmission Type Distribution")
    top_transmissions = train["transmission_simplified"].value_counts().head(5).index
    filtered = train[train["transmission_simplified"].isin(top_transmissions)]

    fig5 = px.histogram(
        filtered,
        x="transmission_simplified",
        title="Top 5 Transmission Types by Count"
    )
    fig5.update_xaxes(title="Transmission Type", categoryorder="total descending")
    fig5.update_yaxes(title="Count")
    st.plotly_chart(fig5, use_container_width=True)

    #Figure 6
    st.subheader("6. Car Price vs. Miles Driven")
    # Filter and sample data]
    sampled = filtered.sample(n=2000, random_state=42)
    # Create scatter plot with trendline
    fig6 = px.scatter(
        sampled,
        x="milage",
        y="price",
        title="Car Price vs. Miles Driven (Sampled)",
        labels={"milage": "Miles Driven", "price": "Price ($)"},
        opacity=0.6,
        trendline="ols"
    )
    # Find and modify the trendline trace
    for trace in fig6.data:
        if trace.mode == "lines":
            trace.line.color = "red"
    # Axis formatting
    fig6.update_yaxes(tickformat="$,", title="Price ($)")
    fig6.update_xaxes(title="Miles Driven")
    st.plotly_chart(fig6, use_container_width=True)


# Tab 3: ML Results
with tab3:
    st.header("Model Results")

    st.subheader("Model Performance Summary")
    st.markdown("""
    We trained multiple machine learning models using 5-fold cross-validation and stacked them for final predictions:

    - **XGBoost Regressor**
    - **LightGBM Regressor**
    - **HistGradientBoostingRegressor (Scikit-Learn)**
    - **AutoGluon FastAI Regressor**
    - **Ridge Stacking Ensemble**

    **Final Ensemble RMSE**: ~71,595
    """)

    st.subheader(" Model Comparison Table")
    model_scores = {
        "Model": ["XGBoost", "LightGBM", "HistGradientBoosting", "AutoGluon (FastAI)", "Stacked Ridge Ensemble"],
        "RMSE": [73410, 72600, 74500, 73000, 71595]
    }
    score_df = pd.DataFrame(model_scores)
    st.dataframe(score_df)

    st.subheader("Top 20 Feature Importances (XGBoost Sampled)")
    sampled = train.sample(frac=0.1, random_state=42)
    y = sampled['price']
    X = sampled.drop(columns=['price'])
    X['is_luxury'] = X['brand'].isin(["BMW", "Mercedes-Benz", "Audi", "Porsche", "Lexus", "Genesis", "Jaguar"]).astype(int)
    cat_cols = ['brand', 'fuel_type', 'transmission_simplified', 'clean_title', 'accident']
    X_encoded = pd.get_dummies(X[cat_cols], drop_first=True)
    X_final = pd.concat([X.select_dtypes(exclude=['object']), X_encoded], axis=1)
    model = XGBRegressor(tree_method='hist', device='cuda', learning_rate=0.05, n_estimators=300)
    model.fit(X_final, y)
    importance_df = pd.DataFrame({
        "Feature": X_final.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(20)

    fig_imp, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=importance_df, y="Feature", x="Importance", palette="viridis", ax=ax)
    ax.set_title("Top 20 Most Important Features (XGBoost)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig_imp)