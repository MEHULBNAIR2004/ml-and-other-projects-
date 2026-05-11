import pandas as pd

import numpy as np


import matplotlib.pyplot as plt

import seaborn as sns

df=pd.read_csv(r"C:\Users\MEHUL B NAIR\OneDrive\Desktop\oort project but we fixed it q\Reviews.csv")


 

df=df.head(100)
 

#quick eda
import nltk
nltk.download('punkt_tab') # Changed from 'punkt' as per error message
from nltk.tokenize import word_tokenize

import re

example=df['Text'][50]


tokens=nltk.word_tokenize(example)
 


 


 




#vader modeel 

from nltk.sentiment import SentimentIntensityAnalyzer


from tqdm import tqdm


sia=SentimentIntensityAnalyzer()


 
j2=sia.polarity_scores('good very good very good very very good ')


res={}

for i,row in tqdm(df.iterrows(),total=len(df)):

    text=row['Text']
    myid=row['Id']
    res[myid]=sia.polarity_scores(text)


vader=pd.DataFrame(res).T

vader=vader.reset_index().rename(columns={'index':'Id'})
vader=vader.merge(df,how='left')

 

 #plot vader results 
 

graph=sns.barplot(data=vader,x='Score',y='compound')



graph.set_title('amazon star reviw by vader')

plt.show()


fig,axs=plt.subplots(1,3,figsize=(12,3))

sns.barplot(data=vader,x="Score",y='pos',ax=axs[0])

sns.barplot(data=vader,x="Score",y='neg',ax=axs[1])
sns.barplot(data=vader,x="Score",y='neu',ax=axs[2])
axs[0].set_title('positive shit ')
axs[1].set_title('negative shit ')
axs[2].set_title('neutral shit  shit ')
plt.show()


 

  