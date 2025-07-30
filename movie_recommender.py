import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv('C:/Users/user/Downloads/ml-latest-small/ml-latest-small/movies.csv')
tags = pd.read_csv('C:/Users/user/Downloads/ml-latest-small/ml-latest-small/tags.csv')

# Preprocess
movies['genres'] = movies['genres'].fillna('')
tags['tag'] = tags['tag'].fillna('').str.lower()
movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
movies = movies.merge(movie_tags, on='movieId', how='left')
movies['combined_features'] = movies['genres'] + " " + movies['tag'].fillna('')

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend function
def recommend_movies(title):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    return movies.iloc[[i[0] for i in sim_scores[1:11]]]

# UI
st.title("🎬 Smart Movie Recommender")
selected_movie = st.selectbox("Choose a movie", movies['title'].values)

if st.button("Recommend"):
    results = recommend_movies(selected_movie)
    st.write("### Recommended Movies:")
    for _, row in results.iterrows():
        st.markdown(f"**{row['title']}**")
        st.write(row['genres'])
        st.write("---")

        #streamlit run movie_recommender.py