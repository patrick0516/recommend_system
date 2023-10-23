from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from fuzzywuzzy import fuzz
from flask import Flask, request, render_template

app = Flask(__name__)

# 讀取資料
df = pd.read_csv("youtube.csv", encoding="latin-1")

## 選擇模型需要的特徵與屬性
features = ['Youtuber', 'rank', 'subscribers', 'video views', 'category', 'uploads', 'Country', 'channel_type', 'Title']

## 選擇其中四個做成一列
def combine_features(row):
    return row['Youtuber'] + " " + row['Title'] + " " + row['Country'] + " " + row['channel_type']

## 有空值會影響模型的準確度，所以空值填入空字串 
for feature in features:
    df[feature] = df[feature].fillna('')  # 填充所有NaN為空字串
df["combined_features"] = df.apply(combine_features, axis=1)

## 當你將文本數據通過「count vectorizer」函數處理時，它會返回一個矩陣，其中包含每個單詞在每個文檔中出現的次數
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

## 使用Cosine Similarity餘弦相似性來計算相似度，其特徵是不受文本長度的影響，只受文本內容的影響，因此比較適合用來做文本相似度的比較
cosine_sim = cosine_similarity(count_matrix)

## 使用fuzzywuzzy套件來找出與使用者輸入最相似的Youtuber
def find_similar_youtuber(user_input, youtuber_list):
    best_match = None
    highest_similarity = -1
    
    for youtuber in youtuber_list:
        similarity = fuzz.ratio(user_input.lower(), youtuber.lower())
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = youtuber
    
    return best_match

## 找出使用者輸入的Youtuber在資料集中的index
def get_index_from_title(Youtuber):
    return df[df.Youtuber == Youtuber].index[0]

def get_title_from_index(index):
    return df[df.index == index]["Youtuber"].values[0]

def find_youtubers_by_category(category):
    return df[df['category'] == category].sort_values(by='video views', ascending=False).head(10)

## 找出與使用者輸入的Youtuber最相似的Youtuber，並列出前十名
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

## 使用Flask來建立網頁，並連接到前端
@app.route('/')
def index():
    return render_template('category_selector.html')

@app.route('/recommend', methods=['GET'])
def recommend_youtubers():
    category = request.args.get('category')
    closest_youtubers = find_similar_youtuber_by_attribute(category)

    if closest_youtubers:
        return "<br>".join(closest_youtubers)
    else:
        return f"找不到與'{category}'相似的Youtuber。請再試一次。"

if __name__ == '__main__':
    app.run(debug=True)
