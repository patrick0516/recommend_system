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

def recommend_youtubers():
    youtube_user_likes = input("輸入一位Youtuber的名字: ")
    closest_youtuber = find_similar_youtuber(youtube_user_likes, df['Youtuber'])

    if closest_youtuber:
        youtube_index = get_index_from_title(closest_youtuber)
        similar_youtubers = list(enumerate(cosine_sim[youtube_index]))

        sorted_similar_youtubers = sorted(similar_youtubers, key=lambda x: x[1], reverse=True)[1:]

        i = 0
        print(f"與'{closest_youtuber}'最相似的前10位Youtuber:\n")
        for element in sorted_similar_youtubers:
            print(get_title_from_index(element[0]))
            i = i + 1
            if i > 10:
                break
    else:
        print(f"找不到與'{youtube_user_likes}'相似的Youtuber。請再試一次。")

def recommend_by_views():
    category = input("請輸入類別: ")
    youtubers = find_youtubers_by_category(category)
    print(f"\n類別為'{category}'的前10名Youtuber按照觀看次數排序如下：\n")
    print(youtubers['Youtuber'].values)

recommendation_type = input("你想要找相似的Youtuber還是按照觀看次數進行推薦？(相似/觀看次數) ")

if recommendation_type == '相似':
    recommend_youtubers()
elif recommendation_type == '觀看次數':
    recommend_by_views()
else:
    print("請輸入有效的選擇 (相似/觀看次數)")
