import os

print("Files in current directory:")
for file in os.listdir('.'):
    if file.endswith('.csv'):
        print(f"  📁 {file}")
        
# Check if our file exists
target_files = [

    'i_beruashvili25_38765.csv',
    'spam_data.csv'
]

print("\nLooking for data files:")
for file in target_files:
    if os.path.exists(file):
        print(f"✅ FOUND: {file}")
    else:
        print(f"❌ MISSING: {file}")