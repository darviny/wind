#!/usr/bin/env python3
"""
extract_acf_features.py - Extract basic statistical features from vibration sensor data.

This script processes time-series sensor data and computes:
- Mean and standard deviation for each sensor
using overlapping windows with 50% overlap.

USAGE:
    python extract_acf_features.py --input data/sensor_data.csv --output data/features.csv --fs 5 --window 2

ARGUMENTS:
    --input: Path to input CSV file containing sensor data (required)
    --output: Path to output CSV file for extracted features (default: data/sensor_features.csv)
    --fs: Sampling frequency in Hz (default: 4)
    --window: Window size in seconds (default: 2)

EXAMPLE:
    # Extract features from sensor data with 5Hz sampling rate and 2-second windows
    python extract_acf_features.py --input data/sensor_data.csv --output data/features.csv --fs 5 --window 2
    
    # Extract features with default parameters
    python extract_acf_features.py --input data/sensor_data.csv

INPUT FORMAT:
    The input CSV should contain columns for sensor readings:
    - timestamp: ISO format timestamp
    - accel_x, accel_y, accel_z: Accelerometer readings
    - gyro_x, gyro_y, gyro_z: Gyroscope readings
    - temperature: Temperature reading (will be dropped)

OUTPUT FORMAT:
    The output CSV will contain features for each window:
    - For each sensor (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z):
        - mean: Average value
        - std: Standard deviation
"""

import pandas as pd
import numpy as np
import argparse
from typing import List, Tuple, Generator, Dict

def load_sensor_data(filepath: str) -> pd.DataFrame:
    """
    Load and prepare sensor data from CSV file.
    
    Args:
        filepath: Path to input CSV file
    
    Returns:
        DataFrame with numeric sensor columns only
    """
    # Define column names
    columns = [
        'timestamp',
        'accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'temperature'  # Keep in loading but will drop later
    ]
    
    print("\nLoading data from: " + filepath)
    
    # Read CSV without headers
    df = pd.read_csv(filepath, names=columns)
    print("Initial data shape: " + str(df.shape))
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print("\nDropping rows with missing values:")
        print(missing[missing > 0])
        df = df.dropna()
        print("Remaining rows: " + str(len(df)))
    
    # Drop timestamp and temperature columns
    numeric_df = df.drop(['timestamp', 'temperature'], axis=1)
    print("\nNumeric columns: " + str(numeric_df.columns.tolist()))
    
    return numeric_df

def generate_windows(data: np.ndarray, 
                    window_size: int, 
                    step_size: int) -> Generator[np.ndarray, None, None]:
    """
    Generate overlapping windows from the input data.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        window_size: Number of samples per window
        step_size: Number of samples to advance each step
    
    Yields:
        Windows of shape (window_size, n_features)
    """
    n_samples = len(data)
    
    for start in range(0, n_samples - window_size + 1, step_size):
        yield data[start:start + window_size]

def compute_basic_features(signal: np.ndarray) -> List[float]:
    """
    Compute basic statistical features for a signal.
    
    Args:
        signal: Input signal array
    
    Returns:
        List of basic features (mean, std)
    """
    return [
        np.mean(signal),  # mean
        np.std(signal)    # std
    ]

def process_data(data: pd.DataFrame, 
                fs: int, 
                buffer_seconds: int) -> pd.DataFrame:
    """
    Process sensor data and extract basic statistical features.
    
    Args:
        data: Input DataFrame with sensor readings
        fs: Sampling frequency in Hz
        buffer_seconds: Window size in seconds
    
    Returns:
        DataFrame containing all computed features
    """
    # Convert DataFrame to numpy array
    data_array = data.values
    
    # Calculate window parameters
    window_size = fs * buffer_seconds
    step_size = window_size // 2  # 50% overlap
    
    print("\nProcessing parameters:")
    print("Sampling rate: " + str(fs) + " Hz")
    print("Window size: " + str(window_size) + " samples (" + str(buffer_seconds) + " seconds)")
    print("Step size: " + str(step_size) + " samples (50% overlap)")
    
    # Process windows
    all_features = []
    window_count = 0
    
    for window in generate_windows(data_array, window_size, step_size):
        # Initialize features list for this window
        window_features = []
        
        # For each sensor (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
        for i in range(6):  # 6 sensors
            sensor_data = window[:, i]
            
            # Add basic features
            window_features.extend(compute_basic_features(sensor_data))
        
        all_features.append(window_features)
        
        window_count += 1
        if window_count % 100 == 0:
            print("Processed " + str(window_count) + " windows...")
    
    print("\nTotal windows processed: " + str(window_count))
    
    # Create column names for the features
    feature_names = []
    for sensor in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        feature_names.extend([
            sensor + '_mean',
            sensor + '_std'
        ])
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(all_features, columns=feature_names)
    print("Feature matrix shape: " + str(feature_df.shape))
    
    # Print feature names for the first window
    if window_count > 0:
        print("\nFeature names:")
        print(feature_df.columns.tolist())
    
    return feature_df

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Extract basic statistical features from sensor data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract features from sensor data with 5Hz sampling rate and 2-second windows
  python extract_acf_features.py --input data/sensor_data.csv --output data/features.csv --fs 5 --window 2
  
  # Extract features with default parameters
  python extract_acf_features.py --input data/sensor_data.csv
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input CSV file containing sensor data'
    )
    
    parser.add_argument(
        '--output',
        default='data/sensor_features.csv',
        help='Output CSV file (default: data/sensor_features.csv)'
    )
    
    parser.add_argument(
        '--fs',
        type=int,
        default=4,
        help='Sampling frequency in Hz (default: 4)'
    )
    
    parser.add_argument(
        '--window',
        type=int,
        default=2,
        help='Window size in seconds (default: 2)'
    )
    
    args = parser.parse_args()
    
    print("Starting feature extraction...")
    
    try:
        # Load data
        data = load_sensor_data(args.input)
        
        # Process data
        features = process_data(
            data,
            fs=args.fs,
            buffer_seconds=args.window
        )
        
        # Save features
        features.to_csv(args.output, index=False)
        print("\nFeatures saved to: " + args.output)
        
    except Exception as e:
        print("Error processing data: " + str(e))
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 