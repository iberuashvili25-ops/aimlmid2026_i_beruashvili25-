import pandas as pd

print("=== CHECKING YOUR CSV FILE ===")
print("Opening your CSV file...")

# Read your file
df = pd.read_csv('i_beruashvili25_38765.csv')

print(f"\n1. File has {len(df)} rows and {len(df.columns)} columns")
print(f"2. Column names: {df.columns.tolist()}")
print(f"\n3. First 3 rows of data:")
print(df.head(3))
print(f"\n4. Last column (should be spam/legit): '{df.columns[-1]}'")
print(f"   Unique values: {df.iloc[:, -1].unique()}")
print(f"\n5. Spam vs Legit counts:")
print(df.iloc[:, -1].value_counts())

print("\n✅ Done! Send me this output.")