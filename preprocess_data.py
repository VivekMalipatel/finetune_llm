import pandas as pd
import csv

df = pd.read_csv('labeled_emails.csv')

df = df[['MessageID','From','To','Subject','Body','Date','Label']]
