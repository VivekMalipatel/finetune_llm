import pandas as pd

# Step 1: Read the file with its original encoding
# Replace 'original_encoding' with the encoding of your file.
# For example, if you've identified the file is in 'MacRoman', use 'mac_roman'
file_path = 'preprocessed_emails_balanced.csv'  # Path to your file
original_encoding = 'mac_roman'  # Replace with the actual encoding of the file
df = pd.read_csv(file_path, encoding=original_encoding)

# Step 2: Save the file back with UTF-8 encoding
utf8_file_path = 'preprocessed_emails_balanced_utf8.csv'  # You can choose a new name or overwrite the original file
df.to_csv(utf8_file_path, index=False, encoding='utf-8')

print(f"File has been converted to UTF-8 and saved as: {utf8_file_path}")
