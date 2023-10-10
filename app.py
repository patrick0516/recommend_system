import pickle
import streamlit as st
import requests


def recommend(youtube):
    index = youtube_df[youtube_df['Title'] == youtube].index[0]
    distances = sorted(list(enumerate(similary[index])), reverse=True, key = lambda x: x[1])
    recommended_youtube_name = []
    recommended_youtube_poster = []
    for i in distances[1:6]:
        youtube_id = youtube_df.iloc[i[0]]['channel_type']
        recommended_youtube_poster.append(fetch_poster(youtube_id))
        recommended_youtube_name.append(youtube_df.iloc[i[0]].title)
    return recommended_youtube_name, recommended_youtube_poster


st.header('Video Recommender System')
youtubes = pickle.load(open('artificats/youtube_list.pkl', 'rb'))
similarity = pickle.load(open('artificats/similarity_list.pkl', 'rb'))

youtube_list = youtubes['Title'].values
selected_youtube = st.selectbox('Select a video', youtube_list)

if st.button('Show recommendation'):
    recommended_youtube_name, recommended_youtube_poster = recommend(selected_youtube)
