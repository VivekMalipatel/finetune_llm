import chardet    

file_path = 'preprocessed_emails_balanced.csv'

# Detect the encoding
with open(file_path, 'rb') as file:
    print(chardet.detect(file.read()))