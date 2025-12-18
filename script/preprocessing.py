"""
Preprocessing Script for E-commerce Grey Market Data
This script performs data cleaning, transformation, and feature engineering
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path='../ecommerce_grey_market_data.csv'):
    """Load the raw CSV data"""
    print("=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)
    df = pd.read_csv(file_path)
    print(f"Initial data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    return df

def clean_numeric_column(series, remove_negative=False):
    """Clean numeric columns by removing commas and converting to float"""
    def clean_value(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val) if val >= 0 or not remove_negative else np.nan
        # Convert to string and clean
        val_str = str(val).strip()
        # Remove commas and other non-numeric characters except decimal point and minus
        val_str = re.sub(r'[^\d\.\-]', '', val_str)
        try:
            num_val = float(val_str)
            if remove_negative and num_val < 0:
                return np.nan
            return num_val
        except:
            return np.nan
    
    return series.apply(clean_value)

def extract_rating_from_text(text):
    """Extract numeric rating from text like '4.5 out of 5 stars'"""
    if pd.isna(text):
        return np.nan
    text_str = str(text)
    # Look for patterns like "4.5", "4 out of 5", etc.
    match = re.search(r'(\d+\.?\d*)', text_str)
    if match:
        rating = float(match.group(1))
        # If rating > 5, might be out of 5, so divide
        if rating > 5:
            rating = rating / 10 if rating > 10 else rating / 2
        return rating if 0 <= rating <= 5 else np.nan
    return np.nan

def clean_num_ratings(series):
    """Clean num_ratings column - remove commas and extract numbers"""
    def clean_rating(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val)
        # Remove commas and extract first number
        val_str = re.sub(r',', '', val_str)
        # Remove negative sign if present
        val_str = val_str.replace('-', '')
        match = re.search(r'(\d+)', val_str)
        if match:
            return int(match.group(1))
        return np.nan
    
    return series.apply(clean_rating)

def preprocess_data(df):
    """Main preprocessing function"""
    print("\n" + "=" * 60)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 60)
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # 2.1 Clean Price column
    print("\n2.1 Cleaning Price column...")
    print(f"Price - Before: Missing={df_clean['price'].isna().sum()}, Negative={((df_clean['price'].astype(str).str.contains('-', na=False))).sum()}")
    df_clean['price'] = clean_numeric_column(df_clean['price'], remove_negative=True)
    print(f"Price - After: Missing={df_clean['price'].isna().sum()}, Mean={df_clean['price'].mean():.2f}")
    
    # 2.2 Clean MRP column
    print("\n2.2 Cleaning MRP column...")
    print(f"MRP - Before: Missing={df_clean['mrp'].isna().sum()}")
    df_clean['mrp'] = clean_numeric_column(df_clean['mrp'], remove_negative=True)
    print(f"MRP - After: Missing={df_clean['mrp'].isna().sum()}, Mean={df_clean['mrp'].mean():.2f}")
    
    # 2.3 Calculate discount_percent if missing
    print("\n2.3 Calculating discount_percent...")
    mask = df_clean['discount_percent'].isna() & df_clean['price'].notna() & df_clean['mrp'].notna()
    df_clean.loc[mask, 'discount_percent'] = ((df_clean.loc[mask, 'mrp'] - df_clean.loc[mask, 'price']) / df_clean.loc[mask, 'mrp'] * 100).round(2)
    print(f"Calculated {mask.sum()} discount_percent values")
    
    # 2.4 Clean product_rating
    print("\n2.4 Cleaning product_rating...")
    print(f"product_rating - Before: Missing={df_clean['product_rating'].isna().sum()}")
    df_clean['product_rating'] = df_clean['product_rating'].apply(extract_rating_from_text)
    print(f"product_rating - After: Missing={df_clean['product_rating'].isna().sum()}, Mean={df_clean['product_rating'].mean():.2f}")
    
    # 2.5 Clean num_ratings
    print("\n2.5 Cleaning num_ratings...")
    print(f"num_ratings - Before: Missing={df_clean['num_ratings'].isna().sum()}")
    df_clean['num_ratings'] = clean_num_ratings(df_clean['num_ratings'])
    print(f"num_ratings - After: Missing={df_clean['num_ratings'].isna().sum()}, Mean={df_clean['num_ratings'].mean():.2f}")
    
    # 2.6 Clean review_rating
    print("\n2.6 Cleaning review_rating...")
    df_clean['review_rating'] = df_clean['review_rating'].apply(extract_rating_from_text)
    
    # 2.7 Handle missing brand - extract from product_name
    print("\n2.7 Handling missing brand values...")
    mask = df_clean['brand'].isna() & df_clean['product_name'].notna()
    df_clean.loc[mask, 'brand'] = df_clean.loc[mask, 'product_name'].str.split().str[0]
    print(f"Extracted {mask.sum()} brand values from product_name")
    
    # 2.8 Create derived features
    print("\n2.8 Creating derived features...")
    
    # Price to MRP ratio (suspicious if too low)
    df_clean['price_mrp_ratio'] = np.where(
        (df_clean['price'].notna()) & (df_clean['mrp'].notna()) & (df_clean['mrp'] > 0),
        df_clean['price'] / df_clean['mrp'],
        np.nan
    )
    
    # Flag suspicious pricing (price < 50% of MRP)
    df_clean['suspicious_pricing'] = (df_clean['price_mrp_ratio'] < 0.5).astype(int)
    
    # Review sentiment indicator (simple: has review text = 1, no review = 0)
    df_clean['has_review'] = df_clean['review_text'].notna().astype(int)
    
    # Rating quality (high rating with many ratings = trustworthy)
    df_clean['rating_quality_score'] = np.where(
        (df_clean['product_rating'].notna()) & (df_clean['num_ratings'].notna()),
        df_clean['product_rating'] * np.log1p(df_clean['num_ratings']),
        np.nan
    )
    
    print("Created features: price_mrp_ratio, suspicious_pricing, has_review, rating_quality_score")
    
    # 2.9 Handle categorical columns
    print("\n2.9 Handling categorical columns...")
    df_clean['platform'] = df_clean['platform'].fillna('Unknown')
    df_clean['seller_name'] = df_clean['seller_name'].fillna('Unknown')
    
    # 2.10 Summary statistics
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"\nFinal data shape: {df_clean.shape}")
    print(f"\nMissing values per column:")
    print(df_clean.isnull().sum())
    print(f"\nData types:")
    print(df_clean.dtypes)
    print(f"\nNumeric columns summary:")
    print(df_clean.select_dtypes(include=[np.number]).describe())
    
    return df_clean

def save_processed_data(df, output_path='../processed_data.csv'):
    """Save processed data to CSV"""
    print("\n" + "=" * 60)
    print("STEP 3: SAVING PROCESSED DATA")
    print("=" * 60)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Load data
    df_raw = load_data()
    
    # Preprocess data
    df_processed = preprocess_data(df_raw)
    
    # Save processed data
    save_processed_data(df_processed)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)

