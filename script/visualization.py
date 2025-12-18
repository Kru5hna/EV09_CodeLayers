"""
Visualization Script for E-commerce Grey Market Data Analysis
This script creates comprehensive visualizations for data exploration and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def load_processed_data(file_path='../processed_data.csv'):
    """Load processed data"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded processed data: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Processed data not found. Loading raw data and preprocessing...")
        from preprocessing import load_data, preprocess_data
        df_raw = load_data('../ecommerce_grey_market_data.csv')
        df = preprocess_data(df_raw)
        df.to_csv(file_path, index=False)
        return df

def create_preprocessing_visualizations(df):
    """Create visualizations showing preprocessing steps and data quality"""
    print("\n" + "=" * 60)
    print("CREATING PREPROCESSING VISUALIZATIONS")
    print("=" * 60)
    
    # Create output directory
    import os
    os.makedirs('../visualizations/preprocessing', exist_ok=True)
    
    # 1. Missing Values Heatmap
    print("\n1. Creating Missing Values Heatmap...")
    plt.figure(figsize=(14, 6))
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        sns.heatmap(df[missing_df.index].isnull(), yticklabels=False, cbar=True, cmap='viridis')
        plt.title('Missing Values Heatmap - Before Preprocessing', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../visualizations/preprocessing/01_missing_values_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved: 01_missing_values_heatmap.png")
    
    # 2. Data Quality Bar Chart
    print("\n2. Creating Data Quality Bar Chart...")
    plt.figure(figsize=(12, 6))
    quality_data = {
        'Complete': (df.notna().sum() / len(df) * 100).values,
        'Missing': (df.isna().sum() / len(df) * 100).values
    }
    quality_df = pd.DataFrame(quality_data, index=df.columns)
    quality_df = quality_df.sort_values('Missing', ascending=False)
    
    x = np.arange(len(quality_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, quality_df['Complete'], width, label='Complete', color='#2ecc71')
    bars2 = ax.bar(x + width/2, quality_df['Missing'], width, label='Missing', color='#e74c3c')
    
    ax.set_xlabel('Columns', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Data Quality: Complete vs Missing Values by Column', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(quality_df.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/preprocessing/02_data_quality_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved: 02_data_quality_bar.png")
    
    # 3. Numeric Columns Distribution (Before Cleaning)
    print("\n3. Creating Numeric Distributions...")
    numeric_cols = ['price', 'mrp', 'discount_percent', 'product_rating', 'num_ratings']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols[:6]):
        if col in df.columns:
            data = df[col].dropna()
            if len(data) > 0:
                axes[idx].hist(data, bins=30, edgecolor='black', alpha=0.7, color='#3498db')
                axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(axis='y', alpha=0.3)
    
    # Remove empty subplots
    for idx in range(len(numeric_cols), 6):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Distribution of Numeric Variables', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../visualizations/preprocessing/03_numeric_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved: 03_numeric_distributions.png")
    
    print("\nPreprocessing visualizations completed!")

def create_analysis_visualizations(df):
    """Create analysis visualizations"""
    print("\n" + "=" * 60)
    print("CREATING ANALYSIS VISUALIZATIONS")
    print("=" * 60)
    
    import os
    os.makedirs('../visualizations/analysis', exist_ok=True)
    
    # 1. Platform Distribution
    print("\n1. Creating Platform Distribution...")
    plt.figure(figsize=(10, 6))
    platform_counts = df['platform'].value_counts()
    colors = sns.color_palette("Set2", len(platform_counts))
    plt.pie(platform_counts.values, labels=platform_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=colors, textprops={'fontsize': 12})
    plt.title('Product Distribution by Platform', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('../visualizations/analysis/01_platform_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved: 01_platform_distribution.png")
    
    # 2. Price Analysis by Platform
    print("\n2. Creating Price Analysis by Platform...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot
    price_data = df[df['price'].notna() & (df['price'] > 0)]
    if len(price_data) > 0:
        sns.boxplot(data=price_data, x='platform', y='price', ax=axes[0])
        axes[0].set_title('Price Distribution by Platform', fontweight='bold')
        axes[0].set_ylabel('Price (₹)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Violin plot
        sns.violinplot(data=price_data, x='platform', y='price', ax=axes[1])
        axes[1].set_title('Price Distribution (Violin Plot) by Platform', fontweight='bold')
        axes[1].set_ylabel('Price (₹)')
        axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('../visualizations/analysis/02_price_by_platform.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved: 02_price_by_platform.png")
    
    # 3. Discount Analysis
    print("\n3. Creating Discount Analysis...")
    discount_data = df[df['discount_percent'].notna() & (df['discount_percent'] >= 0) & (df['discount_percent'] <= 100)]
    if len(discount_data) > 0:
        plt.figure(figsize=(12, 6))
        plt.hist(discount_data['discount_percent'], bins=30, edgecolor='black', alpha=0.7, color='#9b59b6')
        plt.xlabel('Discount Percentage (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Discount Percentages', fontsize=14, fontweight='bold')
        plt.axvline(discount_data['discount_percent'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {discount_data["discount_percent"].mean():.2f}%')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('../visualizations/analysis/03_discount_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved: 03_discount_distribution.png")
    
    # 4. Rating Analysis
    print("\n4. Creating Rating Analysis...")
    rating_data = df[df['product_rating'].notna() & (df['product_rating'] >= 0) & (df['product_rating'] <= 5)]
    if len(rating_data) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Rating distribution
        axes[0].hist(rating_data['product_rating'], bins=20, edgecolor='black', alpha=0.7, color='#f39c12')
        axes[0].set_xlabel('Product Rating', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Product Rating Distribution', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Rating by Platform
        sns.boxplot(data=rating_data, x='platform', y='product_rating', ax=axes[1])
        axes[1].set_title('Product Rating by Platform', fontweight='bold')
        axes[1].set_ylabel('Rating')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('../visualizations/analysis/04_rating_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved: 04_rating_analysis.png")
    
    # 5. Top Brands Analysis
    print("\n5. Creating Top Brands Analysis...")
    brand_counts = df['brand'].value_counts().head(10)
    if len(brand_counts) > 0:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=brand_counts.values, y=brand_counts.index, palette='viridis')
        plt.xlabel('Number of Products', fontsize=12)
        plt.ylabel('Brand', fontsize=12)
        plt.title('Top 10 Brands by Product Count', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../visualizations/analysis/05_top_brands.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved: 05_top_brands.png")
    
    # 6. Suspicious Pricing Analysis
    print("\n6. Creating Suspicious Pricing Analysis...")
    if 'suspicious_pricing' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Count plot
        suspicious_counts = df['suspicious_pricing'].value_counts()
        axes[0].bar(['Normal Pricing', 'Suspicious Pricing'], suspicious_counts.values, 
                   color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Suspicious Pricing Flag Count', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # By Platform
        if len(df[df['suspicious_pricing'] == 1]) > 0:
            suspicious_by_platform = df.groupby(['platform', 'suspicious_pricing']).size().unstack(fill_value=0)
            suspicious_by_platform.plot(kind='bar', stacked=True, ax=axes[1], 
                                       color=['#2ecc71', '#e74c3c'], alpha=0.7)
            axes[1].set_title('Suspicious Pricing by Platform', fontweight='bold')
            axes[1].set_ylabel('Count')
            axes[1].set_xlabel('Platform')
            axes[1].legend(['Normal', 'Suspicious'])
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('../visualizations/analysis/06_suspicious_pricing.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved: 06_suspicious_pricing.png")
    
    # 7. Correlation Heatmap
    print("\n7. Creating Correlation Heatmap...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Numeric Variables', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('../visualizations/analysis/07_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved: 07_correlation_heatmap.png")
    
    # 8. Seller Analysis
    print("\n8. Creating Seller Analysis...")
    top_sellers = df['seller_name'].value_counts().head(10)
    if len(top_sellers) > 0:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_sellers.values, y=top_sellers.index, palette='mako')
        plt.xlabel('Number of Products', fontsize=12)
        plt.ylabel('Seller Name', fontsize=12)
        plt.title('Top 10 Sellers by Product Count', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../visualizations/analysis/08_top_sellers.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved: 08_top_sellers.png")
    
    print("\nAnalysis visualizations completed!")

def create_dashboard_summary():
    """Create a summary visualization combining key metrics"""
    print("\n" + "=" * 60)
    print("CREATING DASHBOARD SUMMARY")
    print("=" * 60)
    
    import os
    os.makedirs('../visualizations/dashboard', exist_ok=True)
    
    df = load_processed_data()
    
    # Create a comprehensive dashboard-style figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Platform Distribution (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    platform_counts = df['platform'].value_counts()
    ax1.pie(platform_counts.values, labels=platform_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Platform Distribution', fontweight='bold', fontsize=12)
    
    # 2. Price Distribution (Top Middle)
    ax2 = fig.add_subplot(gs[0, 1])
    price_data = df[df['price'].notna() & (df['price'] > 0) & (df['price'] < df['price'].quantile(0.95))]
    if len(price_data) > 0:
        ax2.hist(price_data['price'], bins=30, edgecolor='black', alpha=0.7, color='#3498db')
        ax2.set_title('Price Distribution', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Price (₹)')
        ax2.set_ylabel('Frequency')
    
    # 3. Rating Distribution (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    rating_data = df[df['product_rating'].notna() & (df['product_rating'] >= 0) & (df['product_rating'] <= 5)]
    if len(rating_data) > 0:
        ax3.hist(rating_data['product_rating'], bins=20, edgecolor='black', alpha=0.7, color='#f39c12')
        ax3.set_title('Rating Distribution', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Rating')
        ax3.set_ylabel('Frequency')
    
    # 4. Top Brands (Middle Left)
    ax4 = fig.add_subplot(gs[1, 0])
    brand_counts = df['brand'].value_counts().head(8)
    if len(brand_counts) > 0:
        ax4.barh(range(len(brand_counts)), brand_counts.values, color='#2ecc71', alpha=0.7)
        ax4.set_yticks(range(len(brand_counts)))
        ax4.set_yticklabels(brand_counts.index)
        ax4.set_xlabel('Count')
        ax4.set_title('Top 8 Brands', fontweight='bold', fontsize=12)
        ax4.invert_yaxis()
    
    # 5. Price by Platform (Middle)
    ax5 = fig.add_subplot(gs[1, 1])
    price_platform = df[df['price'].notna() & (df['price'] > 0)]
    if len(price_platform) > 0:
        sns.boxplot(data=price_platform, x='platform', y='price', ax=ax5)
        ax5.set_title('Price by Platform', fontweight='bold', fontsize=12)
        ax5.set_ylabel('Price (₹)')
        ax5.tick_params(axis='x', rotation=45)
    
    # 6. Discount Distribution (Middle Right)
    ax6 = fig.add_subplot(gs[1, 2])
    discount_data = df[df['discount_percent'].notna() & (df['discount_percent'] >= 0) & (df['discount_percent'] <= 100)]
    if len(discount_data) > 0:
        ax6.hist(discount_data['discount_percent'], bins=25, edgecolor='black', alpha=0.7, color='#9b59b6')
        ax6.set_title('Discount Distribution', fontweight='bold', fontsize=12)
        ax6.set_xlabel('Discount (%)')
        ax6.set_ylabel('Frequency')
    
    # 7. Suspicious Pricing (Bottom Left)
    ax7 = fig.add_subplot(gs[2, 0])
    if 'suspicious_pricing' in df.columns:
        suspicious_counts = df['suspicious_pricing'].value_counts()
        ax7.bar(['Normal', 'Suspicious'], suspicious_counts.values, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax7.set_ylabel('Count')
        ax7.set_title('Suspicious Pricing Flag', fontweight='bold', fontsize=12)
        ax7.grid(axis='y', alpha=0.3)
    
    # 8. Top Sellers (Bottom Middle)
    ax8 = fig.add_subplot(gs[2, 1])
    top_sellers = df['seller_name'].value_counts().head(8)
    if len(top_sellers) > 0:
        ax8.barh(range(len(top_sellers)), top_sellers.values, color='#e67e22', alpha=0.7)
        ax8.set_yticks(range(len(top_sellers)))
        ax8.set_yticklabels([name[:20] + '...' if len(name) > 20 else name for name in top_sellers.index])
        ax8.set_xlabel('Count')
        ax8.set_title('Top 8 Sellers', fontweight='bold', fontsize=12)
        ax8.invert_yaxis()
    
    # 9. Key Statistics (Bottom Right)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    stats_text = f"""
    KEY STATISTICS
    
    Total Products: {len(df):,}
    Platforms: {df['platform'].nunique()}
    Brands: {df['brand'].nunique()}
    Sellers: {df['seller_name'].nunique()}
    
    Avg Price: ₹{df['price'].mean():.2f}
    Avg Rating: {df['product_rating'].mean():.2f}/5
    Avg Discount: {df['discount_percent'].mean():.2f}%
    
    Products with Reviews: {df['has_review'].sum() if 'has_review' in df.columns else 'N/A'}
    Suspicious Pricing: {df['suspicious_pricing'].sum() if 'suspicious_pricing' in df.columns else 'N/A'}
    """
    ax9.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('E-commerce Grey Market Data - Comprehensive Dashboard', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('../visualizations/dashboard/dashboard_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved: dashboard_summary.png")

if __name__ == "__main__":
    # Load data
    df = load_processed_data()
    
    # Create all visualizations
    create_preprocessing_visualizations(df)
    create_analysis_visualizations(df)
    create_dashboard_summary()
    
    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS COMPLETED!")
    print("=" * 60)
    print("\nVisualizations saved in:")
    print("  - visualizations/preprocessing/")
    print("  - visualizations/analysis/")
    print("  - visualizations/dashboard/")

