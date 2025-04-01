#!/usr/bin/env python3
"""
combine_labeled_data.py - Combine and label sensor data from different operating conditions.

This script combines three CSV files containing wind turbine sensor data:
- Normal operation (label 0)
- Tempered blade anomaly (label 1)
- Gearbox imbalance anomaly (label 2)

The combined dataset is shuffled and saved for machine learning purposes.
"""

import pandas as pd
import os

def load_csv_data(filepath, description):
    """
    Load a CSV file and handle potential errors.
    
    Args:
        filepath (str): Path to the CSV file
        description (str): Description of the data for error messages
    
    Returns:
        pandas.DataFrame or None: Loaded data or None if error occurs
    """
    try:
        if not os.path.exists(filepath):
            print(f"Error: {description} file not found: {filepath}")
            return None
            
        df = pd.read_csv(filepath)
        print(f"\nLoaded {description}:")
        print(f"Shape: {df.shape}")
        print(f"First few timestamps: {df['timestamp'].head().tolist()}")
        return df
        
    except Exception as e:
        print(f"Error loading {description}: {e}")
        return None

def main():
    # Step 1: Load each dataset
    print("Loading datasets...")
    
    normal_data = load_csv_data(
        'sensor_data_normal_aggregated.csv',
        'normal operation'
    )
    
    anomaly1_data = load_csv_data(
        'anomaly_type1_aggregated.csv',
        'tempered blade anomaly'
    )
    
    anomaly2_data = load_csv_data(
        'anomaly_type2_aggregated.csv',
        'gearbox imbalance anomaly'
    )
    
    # Check if all datasets were loaded successfully
    if any(df is None for df in [normal_data, anomaly1_data, anomaly2_data]):
        print("\nError: Failed to load all required datasets.")
        return 1
    
    # Step 2: Add labels
    print("\nAdding condition labels...")
    
    # Label 0: Normal operation
    normal_data['label'] = 0
    normal_data['condition'] = 'normal'
    
    # Label 1: Tempered blade anomaly
    anomaly1_data['label'] = 1
    anomaly1_data['condition'] = 'tempered_blade'
    
    # Label 2: Gearbox imbalance
    anomaly2_data['label'] = 2
    anomaly2_data['condition'] = 'gearbox_imbalance'
    
    # Step 3: Combine datasets
    print("\nCombining datasets...")
    combined_data = pd.concat(
        [normal_data, anomaly1_data, anomaly2_data],
        ignore_index=True
    )
    
    print("\nDataset sizes:")
    print(f"Normal operation: {len(normal_data)} samples")
    print(f"Tempered blade: {len(anomaly1_data)} samples")
    print(f"Gearbox imbalance: {len(anomaly2_data)} samples")
    print(f"Combined total: {len(combined_data)} samples")
    
    # Step 4: Shuffle the dataset
    print("\nShuffling combined dataset...")
    shuffled_data = combined_data.sample(
        frac=1,
        random_state=42  # Fixed seed for reproducibility
    ).reset_index(drop=True)
    
    # Step 5: Save to CSV
    output_file = 'labeled_sensor_data.csv'
    print(f"\nSaving to {output_file}...")
    
    shuffled_data.to_csv(output_file, index=False)
    
    # Print summary
    print("\nLabel distribution in final dataset:")
    print(shuffled_data['condition'].value_counts())
    
    print("\nPreview of combined dataset:")
    preview_cols = ['timestamp', 'accel_x_mean', 'accel_y_mean', 'accel_z_mean', 'label', 'condition']
    print(shuffled_data[preview_cols].head())
    
    print(f"\nSuccessfully created {output_file}")
    return 0

if __name__ == "__main__":
    exit(main()) 