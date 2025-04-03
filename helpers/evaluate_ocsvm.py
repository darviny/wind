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

def load_model(model_path='models/model.pkl'):
    """Load the pre-trained model and scaler."""
    try:
        if not os.path.exists(model_path):
            print("Error: Model file " + model_path + " not found!")
            return None, None
            
        print("Loading model from " + model_path + "...")
        model_dict = joblib.load(model_path)
        
        # Verify dictionary contains required components
        if not all(k in model_dict for k in ['model', 'scaler']):
            print("Error: Model file is missing required components!")
            return None, None
            
        return model_dict['model'], model_dict['scaler']
        
    except Exception as e:
        print("Error loading model: " + str(e))
        return None, None

def load_anomaly_data(filepath):
    """Load and prepare anomaly data from CSV."""
    try:
        if not os.path.exists(filepath):
            print("Error: File " + filepath + " not found!")
            return None
            
        print("Loading anomaly data from " + filepath + "...")
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
        print("Error loading anomaly data: " + str(e))
        return None

def get_feature_columns():
    """Return the complete list of features used in the model."""
    feature_names = []
    for sensor in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        # Statistical features
        feature_names.extend([
            sensor + '_mean',
            sensor + '_std',
            sensor + '_max',
            sensor + '_min',
            sensor + '_median',
            sensor + '_q1',
            sensor + '_q3',
            sensor + '_iqr',
            sensor + '_sum_abs',
            sensor + '_sum_squares'
        ])
        # ACF features
        for i in range(4):
            feature_names.append(sensor + '_acf_lag' + str(i+1))
    
    return feature_names

def prepare_features(df, scaler):
    """Prepare and scale feature matrix."""
    feature_cols = get_feature_columns()
    
    # Verify all columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print("Error: Missing columns: " + ", ".join(missing_cols))
        return None
    
    # Extract and scale features
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    
    # Print feature summary
    print("\nFeature Summary:")
    print("=" * 50)
    print("Autocorrelation features: 24 (4 lags × 6 sensors)")
    print("Statistical features: 60 (10 stats × 6 sensors)")
    print("Total features: " + str(len(feature_cols)))
    print("Feature matrix shape: " + str(X_scaled.shape))
    
    return X_scaled

def evaluate_predictions(predictions, model, X):
    """Print evaluation metrics with interpretation."""
    # Count predictions
    n_normal = sum(predictions == 1)
    n_anomaly = sum(predictions == -1)
    total = len(predictions)
    
    print("\nPrediction Summary:")
    print("=" * 50)
    print("Total samples analyzed: " + str(total))
    print("Classified as normal: " + str(n_normal) + " (" + "{:.1f}".format(n_normal/total*100) + "%)")
    print("Classified as anomaly: " + str(n_anomaly) + " (" + "{:.1f}".format(n_anomaly/total*100) + "%)")
    
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
    print("Mean score: " + "{:.3f}".format(np.mean(decision_scores)))
    print("Min score: " + "{:.3f}".format(np.min(decision_scores)))
    print("Max score: " + "{:.3f}".format(np.max(decision_scores)))
    print("Std dev: " + "{:.3f}".format(np.std(decision_scores)))
    
    # Analyze decision score distribution
    print("\nScore Distribution:")
    print("=" * 50)
    percentiles = np.percentile(decision_scores, [25, 50, 75])
    print("25th percentile: " + "{:.3f}".format(percentiles[0]))
    print("Median: " + "{:.3f}".format(percentiles[1]))
    print("75th percentile: " + "{:.3f}".format(percentiles[2]))
    
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
    print("Strong normal (>0.5): " + str(score_distribution['strong_normal']) + " samples")
    print("Weak normal (0 to 0.5): " + str(score_distribution['weak_normal']) + " samples")
    print("Borderline (-0.5 to 0): " + str(score_distribution['borderline']) + " samples")
    print("Strong anomaly (<-0.5): " + str(score_distribution['strong_anomaly']) + " samples")
    
    return decision_scores

def save_results(df, predictions, scores, output_file='../data/evaluation_results.csv'):
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
        print("\nDetailed results saved to " + output_file)
        
        # Print sample of results (without timestamp column)
        print("\nSample of saved results:")
        display_cols = ['prediction_label', 'decision_score', 'confidence_category']
        print(results_df[display_cols].head())
        
        return True
    except Exception as e:
        print("Error saving results: " + str(e))
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
    
    print("\n" + dataset_name + " Analysis:")
    print("=" * 50)
    print("Total samples: " + str(n_samples))
    print("Classified as normal: " + str(n_normal) + " (" + "{:.1f}".format(n_normal/n_samples*100) + "%)")
    print("Classified as anomaly: " + str(n_anomaly) + " (" + "{:.1f}".format(n_anomaly/n_samples*100) + "%)")
    
    print("\nScore Statistics:")
    print("Mean score: " + "{:.3f}".format(np.mean(scores)))
    print("Median score: " + "{:.3f}".format(np.median(scores)))
    print("Std dev: " + "{:.3f}".format(np.std(scores)))
    
    # Score categories
    categories = {
        'Strong normal (>0.5)': sum(scores > 0.5),
        'Weak normal (0 to 0.5)': sum((scores > 0) & (scores <= 0.5)),
        'Borderline (-0.5 to 0)': sum((scores >= -0.5) & (scores <= 0)),
        'Strong anomaly (<-0.5)': sum(scores < -0.5)
    }
    
    print("\nScore Categories:")
    for cat, count in categories.items():
        print(cat + ": " + str(count) + " (" + "{:.1f}".format(count/n_samples*100) + "%)")
    
    return predictions, scores

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Evaluate One-Class SVM model on normal and anomaly data"
    )
    
    parser.add_argument(
        '--model',
        default='models/model.pkl',
        help='Path to saved model file (default: models/model.pkl)'
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
    save_results(df_normal, normal_pred, normal_scores, '../data/normal_evaluation.csv')
    save_results(df_anomaly1, anom1_pred, anom1_scores, '../data/anomaly1_evaluation.csv')
    save_results(df_anomaly2, anom2_pred, anom2_scores, '../data/anomaly2_evaluation.csv')
    
    print("\nEvaluation completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 