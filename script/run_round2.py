"""
Round 2 Main Script
This script runs preprocessing and visualization steps for Round 2 submission
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 70)
    print("ROUND 2: PREPROCESSING & VISUALIZATION")
    print("=" * 70)
    print("\nThis script will:")
    print("1. Preprocess the raw e-commerce data")
    print("2. Generate comprehensive visualizations")
    print("3. Create dashboard summary")
    print("\n" + "=" * 70 + "\n")
    
    # Step 1: Preprocessing
    print("\n[STEP 1/3] Running Preprocessing...")
    print("-" * 70)
    try:
        from preprocessing import load_data, preprocess_data, save_processed_data
        
        # Load raw data
        df_raw = load_data('../ecommerce_grey_market_data.csv')
        
        # Preprocess
        df_processed = preprocess_data(df_raw)
        
        # Save processed data
        save_processed_data(df_processed, '../processed_data.csv')
        
        print("\n✓ Preprocessing completed successfully!")
    except Exception as e:
        print(f"\n✗ Error in preprocessing: {str(e)}")
        return
    
    # Step 2: Visualizations
    print("\n[STEP 2/3] Generating Visualizations...")
    print("-" * 70)
    try:
        from visualization import (
            load_processed_data, 
            create_preprocessing_visualizations,
            create_analysis_visualizations,
            create_dashboard_summary
        )
        
        # Load processed data
        df = load_processed_data('../processed_data.csv')
        
        # Create visualizations
        create_preprocessing_visualizations(df)
        create_analysis_visualizations(df)
        create_dashboard_summary()
        
        print("\n✓ Visualizations generated successfully!")
    except Exception as e:
        print(f"\n✗ Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Summary
    print("\n[STEP 3/3] Summary")
    print("-" * 70)
    print("\n✓ All Round 2 components completed!")
    print("\nGenerated files:")
    print("  - processed_data.csv (cleaned and processed data)")
    print("  - visualizations/preprocessing/ (preprocessing visualizations)")
    print("  - visualizations/analysis/ (analysis visualizations)")
    print("  - visualizations/dashboard/ (dashboard summary)")
    print("\nNext steps:")
    print("  1. Review the visualizations in the visualizations/ folder")
    print("  2. Create Power BI or Tableau dashboard using processed_data.csv")
    print("  3. Update repository with all components")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

