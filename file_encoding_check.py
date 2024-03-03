import chardet    

file_path = 'newsCorpora.csv'

# Detect the encoding
with open(file_path, 'rb') as file:
    print(chardet.detect(file.read()))