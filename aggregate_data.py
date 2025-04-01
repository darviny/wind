#!/usr/bin/env python3
"""
aggregate_data.py - Aggregates raw sensor data into 1-second windows.

Usage:
    python aggregate_data.py input.csv
"""

import pandas as pd
import numpy as np
import sys
import os

def aggregate_data(input_file):
    """Aggregate sensor data into 1-second windows with basic statistics."""
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: File {input_file} not found!")
            return
            
        # Read data with column names
        print(f"Reading {input_file}...")
        df = pd.read_csv(input_file, names=[
            'timestamp',
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'temperature'
        ])
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Group by 1-second windows and compute stats
        print("Computing 1-second window statistics...")
        aggregated = df.resample('1s').agg({
            'accel_x': ['mean', 'std'],
            'accel_y': ['mean', 'std'],
            'accel_z': ['mean', 'std'],
            'gyro_x': ['mean', 'std'],
            'gyro_y': ['mean', 'std'],
            'gyro_z': ['mean', 'std'],
            'temperature': 'mean'
        })
        
        # Flatten column names
        aggregated.columns = [f"{col[0]}_{col[1]}" for col in aggregated.columns]
        aggregated.reset_index(inplace=True)
        
        # Save output
        output_file = input_file.replace('.csv', '_aggregated.csv')
        print(f"Saving to {output_file}...")
        aggregated.to_csv(output_file, index=False)
        
        print(f"\nDone! Processed {len(df)} readings into {len(aggregated)} windows")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python aggregate_data.py input.csv")
        sys.exit(1)
        
    aggregate_data(sys.argv[1]) 