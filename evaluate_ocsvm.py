#!/usr/bin/env python3
"""
evaluate_ocsvm.py - Evaluate a trained One-Class SVM model on known anomaly data.

Usage:
    python evaluate_ocsvm.py --model model.pkl --anomaly1 anomaly_type1_aggregated.csv --anomaly2 anomaly_type2_aggregated.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import sys
import os
import argparse

def load_model(model_path='model.pkl'):
    """Load the pre-trained model and scaler."""
    try:
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found!")
            return None, None
            
        print(f"Loading model from {model_path}...")
        model_dict = joblib.load(model_path)
        
        # Verify dictionary contains required components
        if not all(k in model_dict for k in ['model', 'scaler']):
            print("Error: Model file is missing required components!")
            return None, None
            
        return model_dict['model'], model_dict['scaler']
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def load_anomaly_data(filepath):
    """Load and prepare anomaly data from CSV."""
    try:
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} not found!")
            return None
            
        print(f"Loading anomaly data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print("\nWarning: Found missing values:")
            print(missing[missing > 0])
            print("Dropping rows with missing values...")
            df = df.dropna()
        
        return df
        
    except Exception as e:
        print(f"Error loading anomaly data: {e}")
        return None

def get_feature_columns():
    """Return the complete list of features used in the model."""
    acf_features = [
        # Accelerometer ACF features
        'accel_x_acf_lag1', 'accel_x_acf_lag2', 'accel_x_acf_lag3', 'accel_x_acf_lag4',
        'accel_y_acf_lag1', 'accel_y_acf_lag2', 'accel_y_acf_lag3', 'accel_y_acf_lag4',
        'accel_z_acf_lag1', 'accel_z_acf_lag2', 'accel_z_acf_lag3', 'accel_z_acf_lag4',
        # Gyroscope ACF features
        'gyro_x_acf_lag1', 'gyro_x_acf_lag2', 'gyro_x_acf_lag3', 'gyro_x_acf_lag4',
        'gyro_y_acf_lag1', 'gyro_y_acf_lag2', 'gyro_y_acf_lag3', 'gyro_y_acf_lag4',
        'gyro_z_acf_lag1', 'gyro_z_acf_lag2', 'gyro_z_acf_lag3', 'gyro_z_acf_lag4'
    ]
    
    statistical_features = [
        # Accelerometer statistical features
        'accel_x_mean', 'accel_x_std', 'accel_x_min', 'accel_x_max', 'accel_x_median', 'accel_x_range',
        'accel_y_mean', 'accel_y_std', 'accel_y_min', 'accel_y_max', 'accel_y_median', 'accel_y_range',
        'accel_z_mean', 'accel_z_std', 'accel_z_min', 'accel_z_max', 'accel_z_median', 'accel_z_range',
        # Gyroscope statistical features
        'gyro_x_mean', 'gyro_x_std', 'gyro_x_min', 'gyro_x_max', 'gyro_x_median', 'gyro_x_range',
        'gyro_y_mean', 'gyro_y_std', 'gyro_y_min', 'gyro_y_max', 'gyro_y_median', 'gyro_y_range',
        'gyro_z_mean', 'gyro_z_std', 'gyro_z_min', 'gyro_z_max', 'gyro_z_median', 'gyro_z_range'
    ]
    
    return acf_features + statistical_features

def prepare_features(df, scaler):
    """Prepare and scale feature matrix."""
    feature_cols = get_feature_columns()
    
    # Verify all columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return None
    
    # Extract and scale features
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    
    # Print feature summary
    print("\nFeature Summary:")
    print("=" * 50)
    print(f"Autocorrelation features: 24 (4 lags × 6 sensors)")
    print(f"Statistical features: 36 (6 stats × 6 sensors)")
    print(f"Total features: {len(feature_cols)}")
    print(f"Feature matrix shape: {X_scaled.shape}")
    
    return X_scaled

def evaluate_predictions(predictions, model, X):
    """Print evaluation metrics with interpretation."""
    # Count predictions
    n_normal = sum(predictions == 1)
    n_anomaly = sum(predictions == -1)
    total = len(predictions)
    
    print("\nPrediction Summary:")
    print("=" * 50)
    print(f"Total samples analyzed: {total}")
    print(f"Classified as normal: {n_normal} ({n_normal/total*100:.1f}%)")
    print(f"Classified as anomaly: {n_anomaly} ({n_anomaly/total*100:.1f}%)")
    
    print("\nInterpretation:")
    print("=" * 50)
    anomaly_rate = n_anomaly/total*100
    if anomaly_rate > 30:
        print("⚠️  High anomaly rate detected! Model might be too sensitive.")
    elif anomaly_rate < 5:
        print("ℹ️  Low anomaly rate. Model might be too permissive.")
    else:
        print("✓ Anomaly rate within expected range.")
    
    # Calculate decision boundary statistics
    decision_scores = model.decision_function(X)
    
    print("\nDecision Score Analysis:")
    print("=" * 50)
    print(f"Mean score: {np.mean(decision_scores):.3f}")
    print(f"Min score: {np.min(decision_scores):.3f}")
    print(f"Max score: {np.max(decision_scores):.3f}")
    print(f"Std dev: {np.std(decision_scores):.3f}")
    
    # Analyze decision score distribution
    print("\nScore Distribution:")
    print("=" * 50)
    percentiles = np.percentile(decision_scores, [25, 50, 75])
    print(f"25th percentile: {percentiles[0]:.3f}")
    print(f"Median: {percentiles[1]:.3f}")
    print(f"75th percentile: {percentiles[2]:.3f}")
    
    # Interpretation of decision scores
    print("\nScore Interpretation:")
    print("=" * 50)
    print("• Negative scores indicate stronger anomaly likelihood")
    print("• Positive scores indicate normal operation likelihood")
    print("• Scores near zero are borderline cases")
    
    score_distribution = {
        'strong_normal': sum(decision_scores > 0.5),
        'weak_normal': sum((decision_scores > 0) & (decision_scores <= 0.5)),
        'borderline': sum((decision_scores >= -0.5) & (decision_scores <= 0)),
        'strong_anomaly': sum(decision_scores < -0.5)
    }
    
    print("\nScore Categories:")
    print(f"Strong normal (>0.5): {score_distribution['strong_normal']} samples")
    print(f"Weak normal (0 to 0.5): {score_distribution['weak_normal']} samples")
    print(f"Borderline (-0.5 to 0): {score_distribution['borderline']} samples")
    print(f"Strong anomaly (<-0.5): {score_distribution['strong_anomaly']} samples")
    
    return decision_scores

def save_results(df, predictions, scores, output_file='evaluation_results.csv'):
    """Save detailed results to CSV with categorization."""
    try:
        # Create a copy of the DataFrame to avoid modifying the original
        results_df = df.copy()
        
        # Add predictions and scores
        results_df['prediction'] = predictions
        results_df['prediction_label'] = results_df['prediction'].map({
            1: 'normal_operation',
            -1: 'potential_anomaly'
        })
        results_df['decision_score'] = scores
        
        # Add score categorization
        results_df['confidence_category'] = pd.cut(
            results_df['decision_score'],
            bins=[-np.inf, -0.5, 0, 0.5, np.inf],
            labels=['strong_anomaly', 'borderline', 'weak_normal', 'strong_normal']
        )
        
        # Add evaluation timestamp
        results_df['evaluation_timestamp'] = pd.Timestamp.now()
        
        # Save to CSV
        results_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to {output_file}")
        
        # Print sample of results (without timestamp column)
        print("\nSample of saved results:")
        display_cols = ['prediction_label', 'decision_score', 'confidence_category']
        print(results_df[display_cols].head())
        
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_dataset_separately(model, X, dataset_name):
    """Analyze predictions for a single dataset."""
    predictions = model.predict(X)
    scores = model.decision_function(X)
    
    n_samples = len(predictions)
    n_normal = sum(predictions == 1)
    n_anomaly = sum(predictions == -1)
    
    print(f"\n{dataset_name} Analysis:")
    print("=" * 50)
    print(f"Total samples: {n_samples}")
    print(f"Classified as normal: {n_normal} ({n_normal/n_samples*100:.1f}%)")
    print(f"Classified as anomaly: {n_anomaly} ({n_anomaly/n_samples*100:.1f}%)")
    
    print("\nScore Statistics:")
    print(f"Mean score: {np.mean(scores):.3f}")
    print(f"Median score: {np.median(scores):.3f}")
    print(f"Std dev: {np.std(scores):.3f}")
    
    # Score categories
    categories = {
        'Strong normal (>0.5)': sum(scores > 0.5),
        'Weak normal (0 to 0.5)': sum((scores > 0) & (scores <= 0.5)),
        'Borderline (-0.5 to 0)': sum((scores >= -0.5) & (scores <= 0)),
        'Strong anomaly (<-0.5)': sum(scores < -0.5)
    }
    
    print("\nScore Categories:")
    for cat, count in categories.items():
        print(f"{cat}: {count} ({count/n_samples*100:.1f}%)")
    
    return predictions, scores

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Evaluate One-Class SVM model on normal and anomaly data"
    )
    
    parser.add_argument(
        '--model',
        default='model.pkl',
        help='Path to saved model file (default: model.pkl)'
    )
    
    parser.add_argument(
        '--normal',
        required=True,
        help='Path to normal operation dataset CSV'
    )
    
    parser.add_argument(
        '--anomaly1',
        required=True,
        help='Path to first anomaly dataset CSV'
    )
    
    parser.add_argument(
        '--anomaly2',
        required=True,
        help='Path to second anomaly dataset CSV'
    )
    
    args = parser.parse_args()
    
    # Load model and scaler
    model, scaler = load_model(args.model)
    if model is None or scaler is None:
        return 1
    
    # Load datasets
    df_normal = load_anomaly_data(args.normal)
    df_anomaly1 = load_anomaly_data(args.anomaly1)
    df_anomaly2 = load_anomaly_data(args.anomaly2)
    
    if df_normal is None or df_anomaly1 is None or df_anomaly2 is None:
        return 1
    
    # Prepare features for each dataset
    X_normal = prepare_features(df_normal, scaler)
    X_anomaly1 = prepare_features(df_anomaly1, scaler)
    X_anomaly2 = prepare_features(df_anomaly2, scaler)
    
    if X_normal is None or X_anomaly1 is None or X_anomaly2 is None:
        return 1
    
    # Analyze each dataset separately
    print("\nAnalyzing each dataset separately...")
    
    normal_pred, normal_scores = analyze_dataset_separately(model, X_normal, "Normal Operation Data")
    anom1_pred, anom1_scores = analyze_dataset_separately(model, X_anomaly1, "Anomaly Type 1 Data")
    anom2_pred, anom2_scores = analyze_dataset_separately(model, X_anomaly2, "Anomaly Type 2 Data")
    
    # Save detailed results for each dataset
    save_results(df_normal, normal_pred, normal_scores, 'normal_evaluation.csv')
    save_results(df_anomaly1, anom1_pred, anom1_scores, 'anomaly1_evaluation.csv')
    save_results(df_anomaly2, anom2_pred, anom2_scores, 'anomaly2_evaluation.csv')
    
    print("\nEvaluation completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 