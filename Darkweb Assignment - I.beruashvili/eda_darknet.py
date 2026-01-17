import pandas as pd
import numpy as np

def analyze_dataset(file_path):
    print(f"Loading dataset: {file_path}")
    df = pd.read_csv(file_path)
    
    print("\n--- Basic Info ---")
    print(df.info())
    
    print("\n--- Column Names ---")
    print(df.columns.tolist())
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    print("\n--- Label 1 Distribution ---")
    if 'Label' in df.columns:
        print(df['Label'].value_counts())
    elif 'Label.1' in df.columns:
        print(df['Label.1'].value_counts())
    
    print("\n--- Label 2 Distribution ---")
    # Often Label.1 or similar is the second label in this dataset
    potential_label2 = [col for col in df.columns if 'Label' in col and col != 'Label']
    for col in potential_label2:
        print(f"\nDistribution for {col}:")
        print(df[col].value_counts())

if __name__ == "__main__":
    analyze_dataset('/home/ubuntu/upload/Darknet.csv')
