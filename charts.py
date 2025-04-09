# charts.py (Plotly Version)

import pandas as pd
import plotly.express as px

# Load data
train = pd.read_csv("train.csv")

# Fill missing values for visualization
train['clean_title'] = train['clean_title'].fillna("Unknown")
train['fuel_type'] = train['fuel_type'].fillna("Unknown")
train['accident'] = train['accident'].fillna("None reported")

# Add car age and mileage per year
train['car_age'] = 2025 - train['model_year']
train['mileage_per_year'] = train['milage'] / train['car_age'].replace(0, 1)

# Idea 1: Price Distribution
fig1 = px.histogram(train, x="price", nbins=50, title="Distribution of Car Prices")
fig1.show()

# Idea 2: Price vs. Car Age
fig2 = px.box(train, x="car_age", y="price", title="Car Price by Car Age")
fig2.show()

# Idea 3: Average Price by Brand (Top 15)
top_brands = train['brand'].value_counts().head(15).index
brand_avg_price = train[train['brand'].isin(top_brands)].groupby('brand')['price'].mean().sort_values(ascending=False).reset_index()
fig3 = px.bar(brand_avg_price, x='brand', y='price', title='Average Price by Top 15 Brands')
fig3.show()

# Idea 4: Fuel Type vs. Price
fig4 = px.box(train, x='fuel_type', y='price', title='Price by Fuel Type')
fig4.show()

# Idea 5: Transmission Type Distribution
train['transmission'] = train['transmission'].fillna("Unknown")
train['transmission_simplified'] = train['transmission'].replace({
    'Transmission w/Dual Shift Mode': 'A/T',
    '10-Speed Automatic': 'A/T',
    '6-Speed M/T': 'M/T',
    '7-Speed A/T': 'A/T',
    'Automatic': 'A/T',
    'Manual': 'M/T'
})
fig5 = px.histogram(train, x='transmission_simplified', title='Distribution of Transmission Types')
fig5.show()
