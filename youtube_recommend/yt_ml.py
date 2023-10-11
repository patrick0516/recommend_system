import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("youtube.csv", encoding = "latin-1")

df = df.rename(columns={'Country': 'country'})


# IMPORTANT 

df1 = df[[ 'Youtuber', 'rank', 'subscribers', 'video views', 'category', 'uploads', 'country', 'channel_type']]

from sklearn.preprocessing import LabelEncoder

# Create label encoders for 'country', 'category' and 'channel_type'
label_encoder_country = LabelEncoder()
label_encoder_category = LabelEncoder()
label_encoder_channel_type = LabelEncoder()


# Apply label encoding to the categorical columns
df1['country_encoded'] = label_encoder_country.fit_transform(df1['country'])
df1['category_encoded'] = label_encoder_category.fit_transform(df1['category'])
df1['channel_type_encoded'] = label_encoder_channel_type.fit_transform(df1['channel_type'])

# Drop the original 'country', 'category' and 'channel_type' column
df1.drop(columns=['country'], inplace=True)
df1.drop(columns=['category'], inplace=True)
df1.drop(columns=['channel_type'], inplace=True)

from sklearn.metrics.pairwise import cosine_similarity
# Create a user-item matrix (excluding the 'Youtuber' column)
user_item_matrix = df1.drop('Youtuber', axis=1).values

# Calculate cosine similarity between Youtubers
similarity_matrix = cosine_similarity(user_item_matrix)

sorted(list(enumerate(similarity_matrix[0])), reverse=True, key=lambda x:x[1])[1:8]

sorted(list(enumerate(similarity_matrix[4])), reverse=True, key=lambda x:x[1])[1:8]

def recommend_youtube(youtube_name):
    yt_index = df1[df1['Youtuber'] == youtube_name].index[0]
    distances = similarity_matrix[yt_index]
    yt_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:9]
    for i in yt_list:
        print(df1.iloc[i[0]].Youtuber)

keyword = input("Enter a Youtuber name: ")
recommend_youtube(keyword)
