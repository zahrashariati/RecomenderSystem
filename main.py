import streamlit as st
import requests
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


st.header("Movie Recommender System")

movies= ["toy story", "run", "dragon", "fight club", "train", "trauma" , "almost famous", "runaway", "dream", "dreamland", "avatar"]

movies = pickle.load(open("movie_dataset.pkl", 'rb'))
model = pickle.load(open("stringy_output.pkl", 'rb'))
movies_date = list( movies['release_date'])
movies_list=list(movies['title'] )
movies_list = [i+' '+str(j.year) for i,j in zip(movies_list,movies_date)]

#making the selectbox
selected = st.multiselect("Select 3 movies from dropdown", movies_list) 

if len(selected) > 3:
    st.error("please select only three movies")

# showing the selected items
elif len(selected) ==1:
    
    col1 , col2 = st.columns([1,3])
    t1 =movies.iloc[movies_list.index(selected[0])]['imdb_id']
    
    url = f"http://www.omdbapi.com/?i={t1}&apikey=d5a77f7b"
    re = requests.get(url).json()
    with col1:
        st.image(re['Poster'], width=100)
    with col2:
        st.write(re['Title']+ "(" + re['Year'] + ")")
        st.caption(f"Genres: {re['Genre']}")

elif len(selected) == 2:
    col1 , col2 , col3 , col4 = st.columns([1,2,1,2])
    t1 =movies.iloc[movies_list.index(selected[0])]['imdb_id']
    t2=movies.iloc[movies_list.index(selected[1])]['imdb_id']
    url = f"http://www.omdbapi.com/?i={t1}&apikey=d5a77f7b"
    re = requests.get(url).json()
    with col1:
        st.image(re['Poster'], width= 100)
    with col2:
        st.write(re['Title']+ "(" + re['Year'] + ")")
        st.caption(f"Genres: {re['Genre']}")

    url = f"http://www.omdbapi.com/?i={t2}&apikey=d5a77f7b"
    re = requests.get(url).json()
    with col3:
        st.image(re['Poster'], width=100)
    with col4:
        st.write(re['Title'] + "(" + re['Year'] + ")")
        st.caption(f"Genres: {re['Genre']}")

elif len(selected) == 3:
    col1 , col2 , col3 , col4, col5 , col6 = st.columns([1,2,1,2,1,2])
    t1 =movies.iloc[movies_list.index(selected[0])]['imdb_id']
    t2=movies.iloc[movies_list.index(selected[1])]['imdb_id']
    t3=movies.iloc[movies_list.index(selected[2])]['imdb_id']
    url = f"http://www.omdbapi.com/?i={t1}&apikey=d5a77f7b"
    re = requests.get(url).json()
    with col1:
        st.image(re['Poster'])
    with col2:
        st.write(re['Title']+ "(" + re['Year'] + ")")
        st.caption(f"Genres: {re['Genre']}")
    
    url = f"http://www.omdbapi.com/?i={t2}&apikey=d5a77f7b"
    re = requests.get(url).json()
    with col3:
        st.image(re['Poster'])
    with col4:
        st.write(re['Title'] + "(" + re['Year'] + ")")
        st.caption(f"Genres: {re['Genre']}")
        
    url = f"http://www.omdbapi.com/?i={t3}&apikey=d5a77f7b"
    re = requests.get(url).json()
    with col5:
        st.image(re['Poster'])
    with col6:
        st.write(re['Title']+ "(" + re['Year'] + ")")
        st.caption(f"Genres: {re['Genre']}")
else:
    pass

st.markdown("<hr>", unsafe_allow_html=True)

#the recommendation function that gets the selected movies and gives top ten recommendations
# def recommend(movie):
#     index=movies[movies['title']==movie].index[0]
#     distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector:vector[1])
#     recommend_movie=[]
#     for i in distance[1:11]:
#         movies_id=movies.iloc[i[0]].id
#         recommend_movie.append(movies.iloc[i[0]].title)
#     return recommend_movie

def make_recomendation(user_indicecs,model_out_put,movies,k,):
    if len(user_indicecs)==0:
        return [0]
    user_choise = model_out_put[user_indicecs]
    mean_user_choise = torch.mean(user_choise, dim=0, keepdim=True)
    cos_similarities = torch.nn.functional.cosine_similarity(model_out_put,mean_user_choise, dim=-1)
    _, top_indices = torch.topk(cos_similarities, k)
    similar_movie_title = movies.iloc[top_indices.flatten()]['imdb_id']
    return similar_movie_title.tolist()

page_number = 0
movies_name = []

def show_movie(movie):
    url = f"http://www.omdbapi.com/?i={movie}&apikey=d5a77f7b"
    re = requests.get(url)
    re = re.json() 
    col1 , col2 = st.columns([1,2])
    with col1:
        st.image(re['Poster'])
    with col2:
        st.subheader(re['Title'])
        st.caption(f"Year: {re['Year']} " +    re['Runtime'])
        st.caption(f"Genres: {re['Genre']}")
        st.caption(f"Director: {re['Director']}") 
        st.caption(f"Cast: {re['Actors']}")
        st.write(re['Plot'])
        st.text(f"Rating: {re['imdbRating']}")
        st.progress(float(re['imdbRating'])/10)

button_next , button_prev = None,None
btnMore = None
movie_name=[]
# recommendation button that activates the recommendation function and shows those ten movies with some information
if st.button("Recommend Me"):
    user_indicecs = [movies_list.index(x) for x in selected]
    print(user_indicecs,selected)
    movie_name = make_recomendation(user_indicecs,model,movies,15+len(selected))
    for i in selected:
        index = movies_list.index(i)
        movie_name.remove(movies.iloc[index]['imdb_id'])
    page_number = 0
    try:  
        st.subheader("Top recommendations")
        for movie in movie_name:
            show_movie(movie)
        #btnMore = st.button('More')
        # if st.button('More'):
        #     print(page_number)
        #     page_number +=1
        #     if(page_number <=4):
        #         print(page_number)
        #         for movie in movie_name[page_number*10:(page_number+1)*10]:
        #             show_movie(movie)
    except:
        pass
    pass
    if len(selected) == 0:
        st.error("Please select atleast one movie")
    

