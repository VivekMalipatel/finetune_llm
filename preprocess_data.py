import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk.stem import WordNetLemmatizer
from nltk import download
from imblearn.over_sampling import RandomOverSampler

import string

# Download necessary NLTK data
download('punkt')
download('stopwords')
download('wordnet')

# Read the CSV file
df = pd.read_csv('labeled_emails.csv')

# Selecting the relevant columns
df = df[['MessageID', 'From', 'To', 'Subject', 'Body', 'Date', 'Label']]

df = df.dropna(subset=['Label'])

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove numbers and special characters, keeping only words
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Remove punctuation
    text = re.sub(f"[{string.punctuation}]", "", text)
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

# Oversampling to balance the dataset
ros = RandomOverSampler(random_state=42)
# Assuming 'Label' is your target and other columns are features, adjust as necessary
X = df.drop('Label', axis=1)  # Features
y = df['Label']  # Target

# The fit_resample function requires numerical input, thus ensure your features are appropriately encoded if necessary
X_resampled, y_resampled = ros.fit_resample(X, y)

# Combine the resampled features and labels back into a DataFrame
# Note: You might need to adjust this if your features were encoded or transformed during oversampling
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['Label'] = y_resampled

# Save the cleaned and balanced DataFrame to a new CSV file
df_resampled.to_csv('preprocessed_emails_balanced.csv', index=False, encoding='utf-8')

