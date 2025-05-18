import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def analyze_predictions(file_path):
    """
    Analyze the predicted steering angles vs true steering angles.
    
    Args:
        file_path (str): Path to the predictions file
    """
    # Read the data
    print(f"Reading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with {len(df)} records")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Make sure required columns are present
    required_cols = ['image_number', 'predicted_angle', 'true_angle']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: File should contain columns: {required_cols}")
        print(f"Found columns: {df.columns.tolist()}")
        return
    
    # Calculate error metrics
    df['error'] = df['predicted_angle'] - df['true_angle']
    df['abs_error'] = np.abs(df['error'])
    df['squared_error'] = df['error'] ** 2
    
    # Calculate statistics
    stats = {
        'mean_error': df['error'].mean(),
        'mean_abs_error': df['abs_error'].mean(),
        'root_mean_squared_error': np.sqrt(df['squared_error'].mean()),
        'max_abs_error': df['abs_error'].max(),
        'min_abs_error': df['abs_error'].min(),
        'std_error': df['error'].std(),
        'r2_score': r2_score(df['true_angle'], df['predicted_angle']),
        'count': len(df),
        'max_true_angle': df['true_angle'].max(),
        'min_true_angle': df['true_angle'].min(),
        'max_pred_angle': df['predicted_angle'].max(),
        'min_pred_angle': df['predicted_angle'].min(),
    }
    
    # Print statistics
    print("\n===== PREDICTION STATISTICS =====")
    print(f"Number of predictions: {stats['count']}")
    print(f"Mean Error: {stats['mean_error']:.4f} degrees")
    print(f"Mean Absolute Error: {stats['mean_abs_error']:.4f} degrees")
    print(f"Root Mean Squared Error: {stats['root_mean_squared_error']:.4f} degrees")
    print(f"Max Absolute Error: {stats['max_abs_error']:.4f} degrees")
    print(f"Min Absolute Error: {stats['min_abs_error']:.4f} degrees")
    print(f"Standard Deviation of Error: {stats['std_error']:.4f} degrees")
    print(f"R² Score: {stats['r2_score']:.4f}")
    print(f"True Angle Range: {stats['min_true_angle']:.2f} to {stats['max_true_angle']:.2f} degrees")
    print(f"Predicted Angle Range: {stats['min_pred_angle']:.2f} to {stats['max_pred_angle']:.2f} degrees")
    
    # Calculate error distribution
    error_bins = [-float('inf'), -10, -5, -2, -1, 0, 1, 2, 5, 10, float('inf')]
    error_labels = ['< -10°', '-10° to -5°', '-5° to -2°', '-2° to -1°', '-1° to 0°', 
                    '0° to 1°', '1° to 2°', '2° to 5°', '5° to 10°', '> 10°']
    df['error_bin'] = pd.cut(df['error'], bins=error_bins, labels=error_labels)
    error_distribution = df['error_bin'].value_counts().sort_index()
    error_percentage = (error_distribution / len(df) * 100).round(2)
    
    print("\n===== ERROR DISTRIBUTION =====")
    for bin_name, count in error_distribution.items():
        print(f"{bin_name}: {count} predictions ({error_percentage[bin_name]}%)")
    
    # Calculate percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    abs_error_percentiles = np.percentile(df['abs_error'], percentiles)
    
    print("\n===== ERROR PERCENTILES =====")
    for p, val in zip(percentiles, abs_error_percentiles):
        print(f"{p}th percentile of absolute error: {val:.4f} degrees")
    
    # Create visualization folder if it doesn't exist
    import os
    os.makedirs('visualization', exist_ok=True)
    
    # Plot 1: Scatter plot of predicted vs true angles
    plt.figure(figsize=(10, 6))
    plt.scatter(df['true_angle'], df['predicted_angle'], alpha=0.5)
    plt.plot([-45, 45], [-45, 45], 'r--')  # Perfect prediction line
    plt.xlabel('True Angle (degrees)')
    plt.ylabel('Predicted Angle (degrees)')
    plt.title('Predicted vs True Steering Angles')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('visualization/pred_vs_true_scatter.png')
    
    # Plot 2: Histogram of errors
    plt.figure(figsize=(12, 6))
    plt.hist(df['error'], bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Error (degrees)')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True)
    plt.savefig('visualization/error_histogram.png')
    
    # Plot 3: Line plot of errors over images
    plt.figure(figsize=(14, 7))
    plt.plot(df['image_number'], df['error'])
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Image Number')
    plt.ylabel('Error (degrees)')
    plt.title('Prediction Error by Image Number')
    plt.grid(True)
    plt.savefig('visualization/error_by_image.png')
    
    # Plot 4: Boxplot of absolute errors
    plt.figure(figsize=(8, 6))
    plt.boxplot(df['abs_error'])
    plt.ylabel('Absolute Error (degrees)')
    plt.title('Boxplot of Absolute Prediction Errors')
    plt.grid(True)
    plt.savefig('visualization/abs_error_boxplot.png')
    
    # Plot 5: Heatmap of errors by true angle range
    # Create bins for true angles
    true_angle_bins = [-float('inf'), -30, -20, -10, -5, 0, 5, 10, 20, 30, float('inf')]
    true_angle_labels = ['< -30°', '-30° to -20°', '-20° to -10°', '-10° to -5°', '-5° to 0°', 
                         '0° to 5°', '5° to 10°', '10° to 20°', '20° to 30°', '> 30°']
    df['true_angle_bin'] = pd.cut(df['true_angle'], bins=true_angle_bins, labels=true_angle_labels)
    
    # Calculate mean absolute error for each true angle bin
    error_by_angle = df.groupby('true_angle_bin')['abs_error'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.bar(error_by_angle['true_angle_bin'], error_by_angle['abs_error'])
    plt.xlabel('True Angle Range')
    plt.ylabel('Mean Absolute Error (degrees)')
    plt.title('Mean Absolute Error by True Angle Range')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualization/error_by_angle_range.png')
    
    print("\nAnalysis completed! Visualizations saved to the 'visualization' folder.")

if __name__ == "__main__":
    file_path = "output/predicted_angles.txt"
    analyze_predictions(file_path)