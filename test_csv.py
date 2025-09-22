#!/usr/bin/env python3
"""
Test script to verify CSV data loading for AgriTech Hub
"""

import pandas as pd
import os

def test_csv_loading():
    """Test if the CSV file can be loaded correctly"""
    csv_path = 'combined_crop_data.csv'
    
    print("🔍 Testing CSV data loading...")
    print(f"Looking for file: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: File '{csv_path}' not found!")
        print("Please ensure the CSV file is in the project root directory.")
        return False
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        print(f"✅ CSV file loaded successfully!")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Check required columns
        required_columns = ['N', 'P', 'K', 'ph', 'rainfall', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        print(f"✅ All required columns present!")
        
        # Show data sample
        print(f"\n📖 Sample data (first 5 rows):")
        print(df[required_columns].head())
        
        # Show unique crops
        unique_crops = df['label'].unique()
        print(f"\n🌾 Available crops ({len(unique_crops)}):")
        for i, crop in enumerate(unique_crops[:10]):  # Show first 10
            print(f"  {i+1}. {crop}")
        if len(unique_crops) > 10:
            print(f"  ... and {len(unique_crops) - 10} more")
        
        # Show data ranges
        print(f"\n📊 Data ranges:")
        for col in ['N', 'P', 'K', 'ph', 'rainfall']:
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"  {col}: {min_val} to {max_val}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False

if __name__ == "__main__":
    print("🌾 AgriTech Hub - CSV Data Test")
    print("=" * 40)
    
    success = test_csv_loading()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ CSV data test passed! Ready to run the application.")
        print("\nNext steps:")
        print("1. Set your GROQ_API_KEY in environment variables")
        print("2. Run: python app.py")
        print("3. Open index.html in your browser")
    else:
        print("❌ CSV data test failed! Please fix the issues above.")
    
    input("\nPress Enter to exit...")
