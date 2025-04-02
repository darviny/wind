#!/usr/bin/env python3
"""
combine_labeled_data.py - Combine and label sensor data from different operating conditions.

Usage:
    python combine_labeled_data.py --normal sensor_data_normal_aggregated.csv \
                                 --anomaly1 anomaly_type1_aggregated.csv \
                                 --anomaly2 anomaly_type2_aggregated.csv \
                                 --output labeled_sensor_data.csv
"""

import pandas as pd
import os
import argparse

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
        print("Number of features:", len(df.columns))
        return df
        
    except Exception as e:
        print(f"Error loading {description}: {e}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Combine sensor data from normal operation and anomaly conditions"
    )
    
    parser.add_argument(
        '--normal',
        required=True,
        help='Path to normal operation data CSV'
    )
    
    parser.add_argument(
        '--anomaly1',
        required=True,
        help='Path to tempered blade anomaly data CSV'
    )
    
    parser.add_argument(
        '--anomaly2',
        required=True,
        help='Path to gearbox imbalance anomaly data CSV'
    )
    
    parser.add_argument(
        '--output',
        default='labeled_sensor_data.csv',
        help='Output file path (default: labeled_sensor_data.csv)'
    )
    
    args = parser.parse_args()
    
    # Step 1: Load each dataset
    print("Loading datasets...")
    
    normal_data = load_csv_data(
        args.normal,
        'normal operation'
    )
    
    anomaly1_data = load_csv_data(
        args.anomaly1,
        'tempered blade anomaly'
    )
    
    anomaly2_data = load_csv_data(
        args.anomaly2,
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
    print(f"\nSaving to {args.output}...")
    shuffled_data.to_csv(args.output, index=False)
    
    # Print summary
    print("\nLabel distribution in final dataset:")
    print(shuffled_data['condition'].value_counts())
    
    print("\nPreview of combined dataset:")
    # Update preview columns to show relevant features
    preview_cols = [
        'accel_x_acf_lag1',
        'accel_x_mean',
        'accel_x_std',
        'gyro_x_acf_lag1',
        'gyro_x_mean',
        'gyro_x_std',
        'label',
        'condition'
    ]
    print(shuffled_data[preview_cols].head())
    
    print(f"\nSuccessfully created {args.output}")
    return 0

if __name__ == "__main__":
    exit(main()) 