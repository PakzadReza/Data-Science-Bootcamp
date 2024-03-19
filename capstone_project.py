# This Program Analysis Amazon's Customers Reviews

import spacy     # Import spaCy NLP module
import pandas as pd # Import Pandas module for data handling
from spacytextblob.spacytextblob import SpacyTextBlob # Import TextBlob for sentiment analysis


nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')


dataframe = pd.read_csv('C:/Users/reza_/CoGrammar/T21/amazon_product_reviews.csv') # Read the data file and print a sample

clean_data = dataframe.dropna(subset=['reviews.text']) # remove all missing values from the column

print(f"\n Below shows 5 random examples from the current dataset: \n {clean_data['reviews.text'].sample(n=5)}")

last_index = clean_data.index[-1] # Identify how many rows are in the dataframe

# Below asks for user input for 2 random number and ensure they are within the data range

num1=int(input(f"\n Enter the 1st random integer number to select a review (The number should be smaller than {last_index}) \n "))
while num1 >= last_index:
    num1 = int(input(f"\n The number should be smaller than {last_index}. Please enter a valid number: \n "))

num2=int(input(f"\n Enter the 2nd random integer number to select a review (The number should be smaller than {last_index}) \n "))
while num2 >= last_index:
    num2 = int(input(f"\n The number should be smaller than {last_index}. Please enter a valid number: \n "))

def sentiment(num): # Functon to calculate the review polarity and sentiment
   
   data_nostop=[] # Assign initial value to data list 

   doc = nlp(clean_data.iloc[num]['reviews.text'])

   for token in doc:                     # loop to remove stop words
      if not token.is_stop:
         data_nostop.append(token.text)

   doc_nostop = ' '.join(data_nostop)    # Convert List to String
   
   # print(doc_nostop)  # Test function to check if data cleaning works fine
   
   doc_nostop = nlp(doc_nostop)          # Process the String with Spacy

   polarity = doc_nostop._.blob.polarity # Implement TextBlob Analysis
   sentiment_score = doc_nostop._.blob.sentiment

   return polarity and sentiment_score

# print the outcome for the two selected indexes
print(f"\nBelow is the 2nd review:\n{clean_data['reviews.text'][num1]}")
print(f"\nThe sentiment score for this review is: {sentiment(num1)}")

print(f"\nBelow is the 2nd review:\n{clean_data['reviews.text'][num2]}")
print(f"\nThe sentiment score for this review is: {sentiment(num2)}")
