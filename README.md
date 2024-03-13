# FinalCapstone

The final Capstone project from my Data Science Skills Bootcamp, 

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


