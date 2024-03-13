# FinalCapstone

The final Capstone project from my Data Science Skills Bootcamp.

---

Firstly import the libraries you are going to need or moght need later on.
Secondly you need to alow the computer to understand and process human language using the inbuilt NLP.
I am using 'en_core_web_md' (medium) as it isnt too large and generally gives better results than sm (small).

    import spacy
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    import spacy
    import os.path as path 
    from textblob import TextBlob

    nlp = spacy.load('en_core_web_md')

Then you need to use pandas to read the file you want to gather data from.
Make sure the file you are reading is saved in the same folder as your Jupyter Notebook file.

df.head() shows you all the collumns but just the first 5 rows, usefull to see what you are working with.

    df = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv')

    df.head()

insert screen shot 

From here we can see which data is useful for us and which isn't, for example we just need the data that is linked with the reviews. So here we can make a new dataframe from the original dataset, only including the columns that have the desired data - 'reviews.text' and 'reviews.title'. 

This new dataframe should be seperate to the main dataset as we don't want to alter the original data. This way we can happily play around with the new dataframe without deleting or changing any of the original data.

    reviews_data = df[['reviews.text', 'reviews.title']]
    reviews_data

screenshot 2 

Now we just have the 2 columns with the data that we are looking at. 

We can remove any NA values from the dataframe by using .dropna - inplace=True and axis=0 

    reviews_data.dropna(inplace=True, axis= 0)

Now we are going to create a function to preprocess the text. 

First we are going to convert it al to lower case using .lower() and strip any trailing white space with .strip().
Then we are going to remove stopwords and punctuation from the text using .is_stop and is_punct.
Finally we want to take the left over words and return them but adding a space with the ' '.join method.

    def preprocess(text):
    
        doc = nlp(text.lower().strip())
        processed = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]

        return ' '.join(processed)

We can use the previous function above to create a new column 'processed.text' by taking our 'reviews.text' column as our input and using the apply.(preprocess) with our function in the brackets for apply. 

    reviews_data['processed.text'] = reviews_data['reviews.text'].apply(preprocess)

Then we simply reviews_data to see the updtated dataframe.
It now has 3 columns with the added processed.text column.

    reviews_data

screenshot 3 

Now we want to create a function for sentiment analysis, one that takes in a product review as an input and predicts its sentiment.

    processed_text = reviews_data['processed.text']

    def sentiment_analysis(processed_text):
        blob = TextBlob(str(processed_text))
        polarity = blob.sentiment.polarity
        return polarity

Now we have a working function that gives a number between -1 and 1. If the number is less than zero that means the sentiment is negative, if we get a number greater than zero the sentiment is positive and if we get a score of 0 then it's sentiment is neutral. 

Now let's add this extra information to our dataset, using a for loop to loop through all the reviews and we add our polarity score to a presaved list - polarity_list. Then we use an if/else statement to determine wether the sentiment was negative, positive or neutral, based on the scores value as stated above. 

Finally lets add new columns to our dataset with the lists that we saved our polarity cvalues and the sentiment.

    polarity_list = []
    sentiment_list = []

    for review in reviews_data['processed.text']:
        polarity_score = sentiment_analysis(review)
        polarity_list.append(polarity_score)
        if polarity_score > 0:
            sentiment_list.append('postive')
        elif polarity_score < 0:
            sentiment_list.append('negative')
        else:
            sentiment_list.append('neutral')
        

    reviews_data['polarity.score'] = polarity_list
    reviews_data['sentiment'] = sentiment_list

Now we can see the added columns to our dataset

    reviews_data

screenshot 4

It would be a great idea now to check some of the reviews and see if the function is working properly and giving us some good results.

To do this lets take a few samples from our data and check we get a similar polarity score and sentiment that we would expect.

    reviews_data.loc[83]

screenshot 5

    reviews_data.loc[973]

screenshot 6

    reviews_data.loc[1345]

screenshot 7 

    reviews_data.loc[1953]

screenshot 8

    reviews_data.loc[2864]

screenshot 9

    reviews_data.loc[3027]

screenshot 10

    reviews_data.loc[3865]

screenshot 11

    reviews_data.loc[4493]

I didnt get any 'neutral' sentiment scores when I took some a random selection so I can manually searching for all polarity scores with a 0. There are 308 rows but this has brought up the first 5 and last 5 in the dataset. 

    reviews_data[reviews_data['polarity.score'] == 0]

screenshot 12 

We also didn't get many 'negative' results so we can do another manual search, here we searched for the 'negative' value in the sentiment column.

Again we have here the first 5 and last 5 in the dataset displayed, but we have 258 rows in total that we could choose from. 

    reviews_data[reviews_data['sentiment'] == 'negative']

screenshot 13
