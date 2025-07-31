import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Generate realistic e-commerce sales data
np.random.seed(42)

# Create sample data
dates = pd.date_range('2023-01-01', '2024-01-31', freq='D')
categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports']
products = {
    'Electronics': ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Camera'],
    'Clothing': ['T-Shirt', 'Jeans', 'Dress', 'Shoes', 'Jacket'],
    'Home & Garden': ['Chair', 'Table', 'Plant', 'Lamp', 'Pillow'],
    'Books': ['Fiction', 'Non-Fiction', 'Textbook', 'Comic', 'Biography'],
    'Sports': ['Ball', 'Racket', 'Weights', 'Mat', 'Shoes']
}

# Generate data
data = []
for date in dates:
    # Simulate seasonal trends
    base_orders = 50 + 20 * np.sin(2 * np.pi * date.dayofyear / 365)
    if date.weekday() >= 5:  # Weekend boost
        base_orders *= 1.2
    
    daily_orders = int(np.random.poisson(base_orders))
    
    for _ in range(daily_orders):
        category = np.random.choice(categories, p=[0.3, 0.25, 0.2, 0.15, 0.1])
        product = np.random.choice(products[category])
        
        # Price varies by category
        price_ranges = {
            'Electronics': (50, 2000),
            'Clothing': (15, 200),
            'Home & Garden': (20, 500),
            'Books': (10, 50),
            'Sports': (25, 300)
        }
        
        price = np.random.uniform(*price_ranges[category])
        quantity = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        
        # Customer demographics
        age = np.random.normal(35, 12)
        age = max(18, min(70, age))
        
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
        city = np.random.choice(cities)
        
        data.append({
            'date': date,
            'category': category,
            'product': product,
            'price': round(price, 2),
            'quantity': quantity,
            'revenue': round(price * quantity, 2),
            'customer_age': int(age),
            'city': city
        })

# Create DataFrame
df = pd.DataFrame(data)
print(f"Dataset created with {len(df)} records")
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# DATA ANALYSIS

# 1. Revenue Analysis
print("\n" + "="*50)
print("REVENUE ANALYSIS")
print("="*50)

# Monthly revenue trend
df['month'] = df['date'].dt.to_period('M')
monthly_revenue = df.groupby('month')['revenue'].sum()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
monthly_revenue.plot(kind='line', marker='o')
plt.title('Monthly Revenue Trend')
plt.xlabel('Month')
plt.ylabel('Revenue ($)')
plt.xticks(rotation=45)

# Category performance
category_revenue = df.groupby('category')['revenue'].sum().sort_values(ascending=False)

plt.subplot(1, 3, 2)
category_revenue.plot(kind='bar', color='skyblue')
plt.title('Revenue by Category')
plt.xlabel('Category')
plt.ylabel('Revenue ($)')
plt.xticks(rotation=45)

# Top products by revenue
top_products = df.groupby(['category', 'product'])['revenue'].sum().sort_values(ascending=False).head(10)

plt.subplot(1, 3, 3)
top_products.plot(kind='barh')
plt.title('Top 10 Products by Revenue')
plt.xlabel('Revenue ($)')

plt.tight_layout()
plt.show()

print(f"Total Revenue: ${df['revenue'].sum():,.2f}")
print(f"Average Order Value: ${df['revenue'].mean():.2f}")
print(f"Total Orders: {len(df):,}")

# 2. Customer Analysis
print("\n" + "="*50)
print("CUSTOMER ANALYSIS")
print("="*50)

# Age group analysis
df['age_group'] = pd.cut(df['customer_age'], 
                        bins=[0, 25, 35, 45, 55, 100], 
                        labels=['18-25', '26-35', '36-45', '46-55', '55+'])

age_analysis = df.groupby('age_group').agg({
    'revenue': ['sum', 'mean', 'count']
}).round(2)

print("Revenue by Age Group:")
print(age_analysis)

# City performance
city_performance = df.groupby('city').agg({
    'revenue': 'sum',
    'quantity': 'sum'
}).sort_values('revenue', ascending=False)

print("\nCity Performance:")
print(city_performance)

# 3. Seasonal Analysis
print("\n" + "="*50)
print("SEASONAL ANALYSIS")
print("="*50)

df['quarter'] = df['date'].dt.quarter
df['day_of_week'] = df['date'].dt.day_name()

quarterly_performance = df.groupby('quarter')['revenue'].sum()
print("Quarterly Revenue:")
print(quarterly_performance)

# Day of week analysis
dow_performance = df.groupby('day_of_week')['revenue'].mean().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
quarterly_performance.plot(kind='bar', color='lightgreen')
plt.title('Revenue by Quarter')
plt.xlabel('Quarter')
plt.ylabel('Revenue ($)')

plt.subplot(1, 2, 2)
dow_performance.plot(kind='bar', color='orange')
plt.title('Average Daily Revenue by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Average Revenue ($)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 4. Key Insights and Recommendations
print("\n" + "="*50)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("="*50)

insights = [
    f"1. Electronics is the top category, generating ${category_revenue.iloc[0]:,.2f} in revenue",
    f"2. {monthly_revenue.idxmax()} was the best performing month with ${monthly_revenue.max():,.2f}",
    f"3. {city_performance.index[0]} is the top performing city with ${city_performance.iloc[0]['revenue']:,.2f}",
    f"4. Weekend sales are typically {((df[df['date'].dt.weekday >= 5]['revenue'].mean() / df[df['date'].dt.weekday < 5]['revenue'].mean() - 1) * 100):.1f}% higher than weekdays",
    f"5. The 26-35 age group represents our core customer base"
]

for insight in insights:
    print(insight)

print("\nRECOMMENDATIONS:")
recommendations = [
    "• Focus marketing budget on Electronics category for maximum ROI",
    "• Increase inventory for weekend sales periods",
    "• Target 26-35 age demographic with personalized campaigns",
    f"• Expand operations in {city_performance.index[0]} market",
    "• Implement seasonal pricing strategies for Q4 holiday season"
]

for rec in recommendations:
    print(rec)

# 5. Statistical Analysis
print("\n" + "="*50)
print("STATISTICAL ANALYSIS")
print("="*50)

# Correlation analysis
numeric_cols = ['price', 'quantity', 'revenue', 'customer_age']
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Price elasticity by category
price_elasticity = df.groupby('category').apply(
    lambda x: x['quantity'].corr(x['price'])
).round(3)

print("Price-Quantity Correlation by Category:")
print(price_elasticity)

print("\n" + "="*50)
print("ANALYSIS COMPLETE")