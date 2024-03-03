import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk.stem import WordNetLemmatizer
from nltk import download

# Download necessary NLTK data
download('punkt')
download('stopwords')
download('wordnet')

# Read the CSV file
df = pd.read_csv('labeled_emails.csv')

# Selecting the relevant columns
df = df[['MessageID', 'From', 'To', 'Subject', 'Body', 'Date', 'Label']]

df = df.dropna(subset=['Label'])

# Define a function to clean the text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize text
    tokens = word_tokenize(text)
    # Lemmatization
    lemmatizer = WordNetLemmatizer()    
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join the words back into one string
    text = ' '.join(tokens)
    return text

# Apply the preprocessing function to the 'Subject' and 'Body' columns
df['Subject'] = df['Subject'].apply(lambda x: preprocess_text(x) if isinstance(x, str) else x)
df['Body'] = df['Body'].apply(lambda x: preprocess_text(x) if isinstance(x, str) else x)

# Save the cleaned DataFrame to a new CSV file, if needed
df.to_csv('preprocessed_emails.csv', index=False)

