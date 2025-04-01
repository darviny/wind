#!/usr/bin/env python3
"""
evaluate_ocsvm.py - Evaluate a trained One-Class SVM model on known anomaly data.

This script:
1. Loads a pre-trained model and scaler
2. Loads and prepares anomaly data
3. Makes predictions and evaluates performance
4. Saves results to a CSV file
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import sys
import os

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

def prepare_features(df, scaler):
    """Prepare and scale feature matrix."""
    # Define feature columns
    feature_cols = [
        'accel_x_mean', 'accel_x_std',
        'accel_y_mean', 'accel_y_std',
        'accel_z_mean', 'accel_z_std',
        'gyro_x_mean', 'gyro_x_std',
        'gyro_y_mean', 'gyro_y_std',
        'gyro_z_mean', 'gyro_z_std'
    ]
    
    # Verify all columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return None
    
    # Extract and scale features
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    
    print(f"\nFeature matrix shape: {X_scaled.shape}")
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

def save_results(df, predictions, scores, output_file='model_evaluation_results.csv'):
    """Save detailed results to CSV with categorization."""
    try:
        # Add predictions and scores
        df['prediction'] = predictions
        df['prediction_label'] = df['prediction'].map({
            1: 'normal_operation',
            -1: 'potential_anomaly'
        })
        df['decision_score'] = scores
        
        # Add score categorization
        df['confidence_category'] = pd.cut(
            df['decision_score'],
            bins=[-np.inf, -0.5, 0, 0.5, np.inf],
            labels=['strong_anomaly', 'borderline', 'weak_normal', 'strong_normal']
        )
        
        # Add timestamp
        df['evaluation_timestamp'] = pd.Timestamp.now()
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to {output_file}")
        
        # Print sample of results
        print("\nSample of saved results:")
        print(df[['timestamp', 'prediction_label', 'decision_score', 'confidence_category']].head())
        
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

def main():
    # Load model and scaler
    model, scaler = load_model()
    if model is None or scaler is None:
        return 1
    
    # Load anomaly datasets
    df1 = load_anomaly_data('anomaly_type1_aggregated.csv')
    df2 = load_anomaly_data('anomaly_type2_aggregated.csv')
    if df1 is None or df2 is None:
        return 1
    
    # Combine datasets
    print("\nCombining datasets...")
    df_combined = pd.concat([df1, df2], ignore_index=True)
    print(f"Combined shape: {df_combined.shape}")
    
    # Prepare features
    X = prepare_features(df_combined, scaler)
    if X is None:
        return 1
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X)
    
    # Evaluate and get decision scores
    decision_scores = evaluate_predictions(predictions, model, X)
    
    # Save detailed results
    if not save_results(df_combined, predictions, decision_scores):
        return 1
    
    print("\nEvaluation completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 