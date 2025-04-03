#!/usr/bin/env python3
import pandas as pd
import os
import sys

def load_csv_data(filepath, condition):
    """Load data from CSV file and add condition label."""
    if not os.path.exists(filepath):
        print("Error: " + condition + " file not found: " + filepath)
        return None
        
    try:
        df = pd.read_csv(filepath)
        df['condition'] = condition
        print("\nLoaded " + condition + ":")
        print("Shape: " + str(df.shape))
        print("Number of features: " + str(len(df.columns)))
        return df
    except Exception as e:
        print("Error loading " + condition + ": " + str(e))
        return None

def main():
    if len(sys.argv) != 4:
        print("Usage: python combine_labeled_data.py anomaly1.csv anomaly2.csv output.csv")
        return 1

    anomaly1_file = sys.argv[1]
    anomaly2_file = sys.argv[2]
    output_file = sys.argv[3]
    
    print("Loading datasets...")
    
    anomaly1_data = load_csv_data(anomaly1_file, 'tempered_blade')
    anomaly2_data = load_csv_data(anomaly2_file, 'gearbox')
    
    if any(df is None for df in [anomaly1_data, anomaly2_data]):
        print("\nError: Failed to load all required datasets.")
        return 1
    
    print("\nAdding condition labels...")
    
    anomaly1_data['label'] = 1
    anomaly2_data['label'] = 2
    
    print("\nCombining datasets...")
    try:
        combined_data = pd.concat(
            [anomaly1_data, anomaly2_data],
            ignore_index=True
        )
    except Exception as e:
        print("Error combining datasets: " + str(e))
        return 1
    
    print("\nDataset sizes:")
    print("Tempered blade: " + str(len(anomaly1_data)) + " samples")
    print("Gearbox: " + str(len(anomaly2_data)) + " samples")
    print("Total: " + str(len(combined_data)) + " samples")
    
    print("\nShuffling combined dataset...")
    try:
        shuffled_data = combined_data.sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)
    except Exception as e:
        print("Error shuffling data: " + str(e))
        return 1
    
    print("\nSaving to " + output_file + "...")
    try:
        shuffled_data.to_csv(output_file, index=False)
    except Exception as e:
        print("Error saving to " + output_file + ": " + str(e))
        return 1
    
    print("\nSuccessfully created " + output_file)
    return 0

if __name__ == "__main__":
    exit(main()) 