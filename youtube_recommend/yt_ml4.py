from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from fuzzywuzzy import fuzz

# 讀取資料
df = pd.read_csv("youtube.csv", encoding="latin-1")

## Recommend top 10 youtubers based on the subscribers and video views
features = ['Youtuber', 'rank', 'subscribers', 'video views', 'category', 'uploads', 'Country', 'channel_type', 'Title']

def combine_features(row):
    return row['Youtuber'] + " " + row['Title'] + " " + row['Country'] + " " + row['channel_type']

for feature in features:
    df[feature] = df[feature].fillna('')  # 填充所有NaN為空字串
df["combined_features"] = df.apply(combine_features, axis=1)

cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

cosine_sim = cosine_similarity(count_matrix)

def find_similar_youtuber(user_input, youtuber_list):
    best_match = None
    highest_similarity = -1
    
    for youtuber in youtuber_list:
        similarity = fuzz.ratio(user_input.lower(), youtuber.lower())
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = youtuber
    
    return best_match

def get_index_from_title(Youtuber):
    return df[df.Youtuber == Youtuber].index[0]

def get_title_from_index(index):
    return df[df.index == index]["Youtuber"].values[0]

def get_title_from_title(Youtuber):
    return df[df.Youtuber == Youtuber]["Title"].values[0]

def find_youtubers_by_category(category):
    return df[df['category'] == category].sort_values(by='video views', ascending=False).head(10)


def find_similar_youtuber_by_attribute(user_attribute):
    closest_youtuber = find_similar_youtuber(user_attribute, df['Youtuber'])

    if closest_youtuber:
        youtube_index = get_index_from_title(closest_youtuber)
        similar_youtubers = list(enumerate(cosine_sim[youtube_index]))

        sorted_similar_youtubers = sorted(similar_youtubers, key=lambda x: x[1], reverse=True)[1:]

        i = 0
        similar_youtubers_list = []
        for element in sorted_similar_youtubers:
            similar_youtubers_list.append(get_title_from_index(element[0]))
            i = i + 1
            if i > 10:
                break

        return similar_youtubers_list
    else:
        return []

def recommend_by_attribute(attribute):
    closest_youtubers = find_similar_youtuber_by_attribute(attribute)

    if closest_youtubers:
        print(f"與'{attribute}'最相似的前10位Youtuber:\n")
        for youtuber in closest_youtubers:
            print(youtuber)
    else:
        print(f"找不到與'{attribute}'相似的Youtuber。請再試一次。")

recommendation_attribute = input("請輸入想要的分類: ")
recommend_by_attribute(recommendation_attribute)
