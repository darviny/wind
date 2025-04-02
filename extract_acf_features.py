#!/usr/bin/env python3
"""
extract_acf_features.py - Extract both autocorrelation and statistical features 
from vibration sensor data using sliding windows.

This script processes time-series sensor data and computes:
- Autocorrelation features (first 4 lags)
- Statistical features (mean, std, min, max, median, range)
using overlapping windows with 50% overlap.
"""

import pandas as pd
import numpy as np
import os
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
    
    print(f"\nLoading data from: {filepath}")
    
    # Read CSV without headers
    df = pd.read_csv(filepath, names=columns)
    print(f"Initial data shape: {df.shape}")
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print("\nDropping rows with missing values:")
        print(missing[missing > 0])
        df = df.dropna()
        print(f"Remaining rows: {len(df)}")
    
    # Drop timestamp and temperature columns
    numeric_df = df.drop(['timestamp', 'temperature'], axis=1)
    print("\nNumeric columns:", numeric_df.columns.tolist())
    
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

def compute_acf_features(window: np.ndarray, n_lags: int = 4) -> Dict[str, float]:
    """
    Compute autocorrelation features for each sensor in a window.
    
    Args:
        window: Array of shape (window_size, n_features)
        n_lags: Number of ACF lags to use as features
    
    Returns:
        Dictionary of ACF features
    """
    features = {}
    
    # Remove temperature from column names
    for col_idx, col_name in enumerate(['accel_x', 'accel_y', 'accel_z',
                                      'gyro_x', 'gyro_y', 'gyro_z']):
        # Extract signal
        signal = window[:, col_idx]
        
        # Remove mean
        signal = signal - np.mean(signal)
        
        # Compute autocorrelation
        acf = np.correlate(signal, signal, mode='full')
        
        # Keep only positive lags
        acf = acf[len(signal)-1:]
        
        # Normalize by zero lag
        if acf[0] > 0:
            acf = acf / acf[0]
        
        # Store first n_lags as features
        for lag in range(1, n_lags + 1):
            features[f'{col_name}_acf_lag{lag}'] = acf[lag]
    
    return features

def compute_aggregate_features(window: np.ndarray) -> Dict[str, float]:
    """
    Compute statistical features for each sensor in a window.
    
    Args:
        window: Array of shape (window_size, n_features)
    
    Returns:
        Dictionary of statistical features
    """
    features = {}
    
    # Remove temperature from column names
    for col_idx, col_name in enumerate(['accel_x', 'accel_y', 'accel_z',
                                      'gyro_x', 'gyro_y', 'gyro_z']):
        # Extract signal
        signal = window[:, col_idx]
        
        # Compute statistics
        features[f'{col_name}_mean'] = np.mean(signal)
        features[f'{col_name}_std'] = np.std(signal)
        features[f'{col_name}_min'] = np.min(signal)
        features[f'{col_name}_max'] = np.max(signal)
        features[f'{col_name}_median'] = np.median(signal)
        features[f'{col_name}_range'] = np.ptp(signal)  # peak-to-peak (max - min)
    
    return features

def process_data(data: pd.DataFrame, 
                fs: int, 
                buffer_seconds: int, 
                n_lags: int = 4) -> pd.DataFrame:
    """
    Process sensor data and extract both ACF and statistical features.
    
    Args:
        data: Input DataFrame with sensor readings
        fs: Sampling frequency in Hz
        buffer_seconds: Window size in seconds
        n_lags: Number of ACF lags to use as features
    
    Returns:
        DataFrame containing all computed features
    """
    # Convert DataFrame to numpy array
    data_array = data.values
    
    # Calculate window parameters
    window_size = fs * buffer_seconds
    step_size = window_size // 2  # 50% overlap
    
    print(f"\nProcessing parameters:")
    print(f"Sampling rate: {fs} Hz")
    print(f"Window size: {window_size} samples ({buffer_seconds} seconds)")
    print(f"Step size: {step_size} samples (50% overlap)")
    print(f"ACF lags per sensor: {n_lags}")
    
    # Process windows
    all_features = []
    window_count = 0
    
    for window in generate_windows(data_array, window_size, step_size):
        # Compute both feature sets
        acf_features = compute_acf_features(window, n_lags)
        agg_features = compute_aggregate_features(window)
        
        # Combine features
        combined_features = {**acf_features, **agg_features}
        all_features.append(combined_features)
        
        window_count += 1
        if window_count % 100 == 0:
            print(f"Processed {window_count} windows...")
    
    print(f"\nTotal windows processed: {window_count}")
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(all_features)
    print(f"Feature matrix shape: {feature_df.shape}")
    
    # Print feature names for the first window
    if window_count > 0:
        print("\nFeature names:")
        print(feature_df.columns.tolist())
    
    return feature_df

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Extract ACF and statistical features from sensor data"
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input CSV file containing sensor data'
    )
    
    parser.add_argument(
        '--output',
        default='sensor_features.csv',
        help='Output CSV file for extracted features'
    )
    
    parser.add_argument(
        '--fs',
        type=int,
        default=4,
        help='Sampling frequency in Hz'
    )
    
    parser.add_argument(
        '--window',
        type=int,
        default=2,
        help='Window size in seconds'
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
        print(f"\nFeatures saved to: {args.output}")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 