import pandas as pd
from ast import literal_eval

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Reader, Dataset, SVD
from surprise.model_selection import GridSearchCV
from surprise.model_selection.validation import cross_validate

import matplotlib as mpl

from helper import calculate_weighted_rating, plot_graph, get_top_tier_recommendation_from_title

credit_small_path = "./dataset/tmdb_5000_credits.csv"

movies_small_path = "./dataset/tmdb_5000_movies.csv"

credit_df = pd.read_csv(credit_small_path)

credit_df = credit_df.drop("title", axis=1)

movie_df = pd.read_csv(movies_small_path)

movie_df = pd.merge(movie_df, credit_df, left_on="id", right_on="movie_id").drop("movie_id", axis=1)

# Demographic Filtering (Find famous movie for new users)
#  Order base on IMDB's weighted rating:

# IMDB's weighted rating formula:
# Weighted Rating(WR) = (v/v+m)*R + (m/v+m)*C

# v = the number of votes for the movie
# m = the minimum votes required
# R = the average rating of the movie
# C = the mean vote across the whole report

C = movie_df['vote_average'].mean()

m = movie_df['vote_count'].quantile(0.9)

# Filter out all the movie which got less than quantile(0.9) = 1838 votes

filtered_movie_df = movie_df[movie_df['vote_count'] > m].copy()

filtered_movie_df['score'] = filtered_movie_df.apply(calculate_weighted_rating, axis=1, args=(C, m))

filtered_movie_df = filtered_movie_df.sort_values('score', ascending=False)

print("Demographic Filtering, base on IMDB's weighted rating: \n", filtered_movie_df[['title', 'vote_count', 'vote_average', 'score']].head(10))

# Demographic Filtering, base on IMDB's weighted rating:
#                                               title  vote_count  vote_average     score
# 1881                       The Shawshank Redemption        8205           8.5  8.059258
# 662                                      Fight Club        9413           8.3  7.939256
# 65                                  The Dark Knight       12002           8.2  7.920020
# 3232                                   Pulp Fiction        8428           8.3  7.904645
# 96                                        Inception       13752           8.1  7.863239
# 3337                                  The Godfather        5893           8.4  7.851236
# 95                                     Interstellar       10867           8.1  7.809479
# 809                                    Forrest Gump        7927           8.2  7.803188
# 329   The Lord of the Rings: The Return of the King        8064           8.1  7.727243
# 1990                        The Empire Strikes Back        5879           8.2  7.697884

# Order base on popularity column
popularity_sorted_df = movie_df.sort_values('popularity', ascending=False)

print("Demographic Filtering, base on popularity: \n", popularity_sorted_df[['title', 'vote_count', 'vote_average', 'popularity']].head(10))

# Demographic Filtering, base on popularity:
#                                                  title  vote_count  vote_average  popularity
# 546                                            Minions        4571           6.4  875.581305
# 95                                        Interstellar       10867           8.1  724.247784
# 788                                           Deadpool       10995           7.4  514.569956
# 94                             Guardians of the Galaxy        9742           7.9  481.098624
# 127                                 Mad Max: Fury Road        9427           7.2  434.278564
# 28                                      Jurassic World        8662           6.5  418.708552
# 199  Pirates of the Caribbean: The Curse of the Bla...        6985           7.5  271.972889
# 82                      Dawn of the Planet of the Apes        4410           7.3  243.791743
# 200              The Hunger Games: Mockingjay - Part 1        5584           6.6  206.227151
# 88                                          Big Hero 6        6135           7.8  203.734590

# Plot graph for Demographic Filtering

attrs = {"super_title": 'Demographic Filtering (Find famous movie for new users)',
        "ax1_movie_title": filtered_movie_df['title'].head(10),
        "ax1_data": filtered_movie_df['score'].head(10),
        "ax1_color": 'tomato',
        "ax1_annotate_label": '{:,.2f}',
        "ax1_title": "Order base on IMDB's weighted rating",
        "ax1_xlabel": "IMDB Weight Score",
        "ax1_set_scale": True,
        "ax1_xaxis_formatter": '{x:.2f}',
        "ax1_xlim": (None, 8.1),

        "ax2_movie_title": popularity_sorted_df['title'].head(10),
        "ax2_data":  popularity_sorted_df['popularity'].head(10),
        "ax2_color": 'tomato',
        "ax2_annotate_label": '{:,.2f}',
        "ax2_title": "Order base on popularity column",
        "ax2_xlabel": "Popularity Score",
        "ax2_set_scale": False,
        "ax2_xaxis_formatter": '{x:.0f}',
        "ax2_xlim": (None, 1100)
        }

plot_graph(attrs)

# Base on user input Genres

movie_df['genres'] = movie_df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

#Pick the top 3 genres
movie_df['genres'] = movie_df['genres'].apply(lambda x: x[0:3] if len(x) > 3 else x)

# user_input = input("Please pick your genres choice(s), seperated by ',' : ")

# user_choices_set = set(user_input.split(","))

user_choices_set = set(['Romance'])

def check_genres(genres_column, user_choices_set):
    if set(genres_column).intersection(user_choices_set) == set():
        return False
    else:
        return True

movie_df['in_genres'] = movie_df['genres'].copy()

movie_df['in_genres'] = movie_df['in_genres'].apply(check_genres, args=(user_choices_set,))

selected_df = movie_df[movie_df['in_genres'] == True]

# Order base on IMDB's weighted rating:

selected_vote_average_mean = selected_df['vote_average'].mean()

selected_C = selected_vote_average_mean

selected_m = selected_df['vote_count'].quantile(0.85)

genres_selected_df = selected_df[selected_df['vote_count'] > selected_m].copy()

genres_selected_df.loc[:,'weighted_rating'] = genres_selected_df.apply(calculate_weighted_rating, axis=1, args=(selected_C, selected_m))

genres_selected_df = genres_selected_df.sort_values('weighted_rating', ascending=False)

print("Base on Genres, order with IMDB's weighted rating: \n", genres_selected_df[['title', 'vote_count', 'vote_average', 'weighted_rating']].head(10))

# Base on Genres, order with IMDB's weighted rating:
#                                       title  vote_count  vote_average   weighted_rating
# 809                            Forrest Gump        7927           8.2          7.995916
# 1997                                    Her        4097           7.9          7.594073
# 2152  Eternal Sunshine of the Spotless Mind        3652           7.9          7.564211
# 2547               The Theory of Everything        3311           7.8          7.458562
# 1260                                 Amélie        3310           7.8          7.458481
# 25                                  Titanic        7562           7.5          7.362073
# 1559                           The Notebook        3067           7.7          7.360417
# 493                        A Beautiful Mind        3009           7.7          7.355387
# 2776        The Perks of Being a Wallflower        2968           7.7          7.351741
# 2838                 The Fault in Our Stars        3759           7.6          7.330217

genres_popularity_sorted_df = genres_selected_df.sort_values('popularity', ascending=False)

# Order base on popularity column
print("Base on Genres, order with popularity: \n", genres_popularity_sorted_df[['title', 'vote_count', 'vote_average', 'popularity']].head(10))

# Base on Genres, order with popularity:
#                        title  vote_count  vote_average  popularity
# 809             Forrest Gump        7927           8.2  138.133331
# 326               Cinderella        2374           6.7  101.187052
# 25                   Titanic        7562           7.5  100.025899
# 1154    Fifty Shades of Grey        3254           5.2   98.755657
# 2313                The Mask        2472           6.6   85.303180
# 1366   The Devil Wears Prada        3088           7.0   83.893257
# 1599      The Age of Adaline        1990           7.4   82.052056
# 2838  The Fault in Our Stars        3759           7.6   74.358971
# 1260                  Amélie        3310           7.8   73.720244
# 3444                  Grease        1581           7.2   67.608041

# Plot grapg for Demographic Filtering base on Genres ('Romance')

attrs = {"super_title": "Demographic Filtering base on Genres ('Romance') (Find famous movie for new users)",
        "ax1_movie_title": genres_selected_df['title'].head(10),
        "ax1_data": genres_selected_df['weighted_rating'].head(10),
        "ax1_color": 'darkcyan',
        "ax1_annotate_label": '{:,.2f}',
        "ax1_title": "Order base on IMDB's weighted rating",
        "ax1_xlabel": "IMDB Weight Score",
        "ax1_set_scale": True,
        "ax1_xaxis_formatter": '{x:.2f}',
        "ax1_xlim": (None, 8.1),

        "ax2_movie_title": genres_popularity_sorted_df['title'].head(10),
        "ax2_data": genres_popularity_sorted_df['popularity'].head(10),
        "ax2_color": 'darkcyan',
        "ax2_annotate_label": '{:,.2f}',
        "ax2_title": "Order base on popularity column",
        "ax2_xlabel": "Popularity Score",
        "ax2_set_scale": False,
        "ax2_xaxis_formatter": '{x:.0f}',
        "ax2_xlim": (None, 160)
        }

plot_graph(attrs)

# Get recommendation base on movie description

# print(movie_df[['overview']].info())    #got 3 nan data

movie_df['overview'] = movie_df['overview'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(movie_df['overview'])

# Since TfidfVectorizer normalized the vector, it can use linear_kernel instead of cosine_similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

title_recomm_df = get_top_tier_recommendation_from_title(movie_df, 'The Dark Knight Rises', cosine_sim, 10)

print("Get recommendation base on movie description: \n", title_recomm_df[['title', 'sim_score']])

# Get recommendation base on movie description:
#                                         title  sim_score
# 65                            The Dark Knight   0.301512 
# 299                            Batman Forever   0.298570 
# 428                            Batman Returns   0.287851 
# 1359                                   Batman   0.264461 
# 3854  Batman: The Dark Knight Returns, Part 2   0.185450 
# 119                             Batman Begins   0.167996 
# 2507                                Slow Burn   0.166829 
# 9          Batman v Superman: Dawn of Justice   0.133740 
# 1181                                      JFK   0.132197 
# 210                            Batman & Robin   0.130455

# Credits, Genres and Keywords Based Recommender

# Data preprocessing for the column 'production_companies', 'genres', 'cast', 'keywords', 'crew', add column 'director'
for feature in ['production_companies', 'cast', 'keywords', 'crew']:
    movie_df[feature] = movie_df[feature].fillna('[]').apply(literal_eval)

    if feature == 'crew':
        movie_df['director'] = movie_df[feature].apply(lambda x: [i['name'] for i in x if (i['job'] == "Director" and isinstance(x, list))])   #every element without director, put []
    else:
        movie_df[feature] = movie_df[feature].apply(lambda x: [i['name'] if isinstance(x, list) else [] for i in x])   #only an element is not a list, put []

for feature in ['production_companies', 'genres', 'cast', 'keywords', 'director']:
    movie_df[feature] = movie_df[feature].apply(lambda x: x[0:3] if len(x) > 3 else x)

    movie_df[feature] = movie_df[feature].apply(lambda x : [str(i).replace(" ","").lower() for i in x])

def combined_feature(x):
    return " ".join(x['genres']) + " " + \
    " ".join(x['cast']) + " " + " ".join(x['keywords']) + " " + \
    " ".join(x['director']) + " " + " ".join(x['production_companies'])

# Add a feature of combining all five column's data
movie_df['combined_feature'] = movie_df.apply(combined_feature, axis = 1)

vectorizer = CountVectorizer(stop_words='english')

count_matrix = vectorizer.fit_transform(movie_df['combined_feature'])

cosine_similarity_matrix = cosine_similarity(count_matrix, count_matrix)

info_df = get_top_tier_recommendation_from_title(movie_df, 'Inside Out', cosine_similarity_matrix, 10)

print("Movie info Based Recommender: \n", info_df[['title', 'sim_score']])
# Movie info Based Recommender: 
#                       title  sim_score
# 231          Monsters, Inc.   0.416667 
# 66                       Up   0.348155 
# 42              Toy Story 3   0.333333 
# 1695                Aladdin   0.333333 
# 4247  Me You and Five Bucks   0.333333 
# 40                   Cars 2   0.320256 
# 118             Ratatouille   0.320256 
# 566                    Cars   0.320256
# 1386       Saving Mr. Banks   0.320256
# 55                    Brave   0.308607

# Plot graph for Content Based Filtering

attrs = {"super_title": 'Content Based Filtering (Recommendation base on movie info)',
        "ax1_movie_title": title_recomm_df['title'],
        "ax1_data": title_recomm_df['sim_score'],
        "ax1_color": 'tab:purple',
        "ax1_annotate_label": '{:,.3f}',
        "ax1_title": "Recommendation base on movie description\n \n \n \n Finding movies similar to 'The Dark Knight Rises'",
        "ax1_xlabel": "Cosine Similarity Score",
        "ax1_set_scale": False,
        "ax1_xaxis_formatter": '{x:.2f}',
        "ax1_xlim": (0.1, 0.35),

        "ax2_movie_title": info_df['title'].head(10),
        "ax2_data": info_df['sim_score'].head(10),
        "ax2_color": 'tab:purple',
        "ax2_annotate_label": '{:,.3f}',
        "ax2_title": "Recommendation base on movie info \n \n ('production_companies', 'genres', 'cast', 'keywords', 'director') \n \n Finding movies similar to 'Inside Out'",
        "ax2_xlabel": "Cosine Similarity Score",
        "ax2_set_scale": False,
        "ax2_xaxis_formatter": '{x:.3f}',
        "ax2_xlim": (None, 0.5)
        }

plot_graph(attrs)

# Collaborative Filtering

reader = Reader()

ratings = pd.read_csv('./dataset/ratings_small.csv')

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# *************Using GridSearchCV to find the hyperparameter******************
# param_grid = {"n_epochs": [5, 10], "lr_all": [0.002, 0.005], "reg_all": [0.4, 0.6]}
# grid_search = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)

# grid_search.fit(data)

# best RMSE score
# print(grid_search.best_score["rmse"])
# 0.9140984221790148

# combination of parameters that gave the best RMSE score
# print(grid_search.best_params["rmse"])
# {'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.4}
# *************End of GridSearchCV to find the hyperparameter*****************

svd = SVD(n_epochs=10, lr_all=0.005, reg_all=0.4)

# cross_validate(svd, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

# ****************** Without hyperparameter turning result******************
#                   Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
# RMSE (testset)    0.8957  0.8886  0.8984  0.9010  0.8961  0.8959  0.0042  
# MAE (testset)     0.6895  0.6823  0.6932  0.6917  0.6921  0.6898  0.0039  
# Fit time          1.39    1.40    1.42    1.43    1.41    1.41    0.01    
# Test time         0.24    0.23    0.24    0.23    0.23    0.23    0.01 
#  ******************Without hyperparameter turning result******************

#  ******************After  hyperparameter turning result******************
#                   Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
# RMSE (testset)    0.9100  0.9083  0.9082  0.9192  0.9059  0.9103  0.0046  
# MAE (testset)     0.7062  0.7077  0.7048  0.7099  0.7055  0.7068  0.0018  
# Fit time          0.74    0.77    0.77    0.79    0.76    0.76    0.02    
# Test time         0.25    0.24    0.23    0.25    0.24    0.24    0.01
#  ******************After  hyperparameter turning result******************

trainset = data.build_full_trainset()
svd.fit(trainset)

# Predict the rating of movie ID 19995 ('Avator') for user 1
print(svd.predict(1, 19995))
# user: 1          item: 19995      r_ui = None   est = 3.00   {'was_impossible': False}

# Hybrid Recommendation

links_df = pd.read_csv('./dataset/links_small.csv')

links_df.rename(columns={'tmdbId': 'id'}, inplace=True)

links_df = links_df.merge(movie_df[['title', 'id']], on='id')

indices_df = links_df.set_index('id')

def get_hybrid_recommendation(userId, title, cosine_similarity_matrix):

    top_tier_df = get_top_tier_recommendation_from_title(movie_df, title, cosine_similarity_matrix, 30)

    rem_df = top_tier_df[['title', 'vote_count', 'vote_average', 'id']].copy()

    rem_df['est'] = rem_df['id'].apply(lambda x: svd.predict(userId, indices_df.loc[x]['movieId']).est if x in indices_df.index else 0)

    rem_df = rem_df.sort_values('est', ascending=False)

    return rem_df.head(10)

hybrid_recom_user_1 = get_hybrid_recommendation(1, 'Avatar', cosine_similarity_matrix)
hybrid_recom_user_500 = get_hybrid_recommendation(500, 'Avatar', cosine_similarity_matrix)

print("Hybrid_recommendation for userID 1 is: \n", hybrid_recom_user_1)
print("Hybrid_recommendation for userID 500 is: \n", hybrid_recom_user_500)

# Hybrid_recommendation for userID 1 is: 
#                                                   title  vote_count  vote_average      id       est
# 101                                  X-Men: First Class        5181           7.1   49538  3.268708
# 2403                                             Aliens        3220           7.7     679  3.258201
# 216                                          Life of Pi        5797           7.2   87827  3.244867
# 46                           X-Men: Days of Future Past        6032           7.5  127585  3.156787
# 989                                      Baby's Day Out         274           5.8   11212  3.095684
# 341   Percy Jackson & the Olympians: The Lightning T...        2010           6.0   32657  3.079153
# 587                                           The Abyss         808           7.1    2756  3.076313
# 261                               Live Free or Die Hard        2089           6.4    1571  3.028606
# 1542                                              Speed        1783           6.8    1637  3.021255
# 33                                X-Men: The Last Stand        3525           6.3   36668  2.996783

# Hybrid_recommendation for userID 500 is:
#                                                   title  vote_count  vote_average      id       est
# 101                                  X-Men: First Class        5181           7.1   49538  3.471333
# 2403                                             Aliens        3220           7.7     679  3.455121
# 216                                          Life of Pi        5797           7.2   87827  3.373537
# 46                           X-Men: Days of Future Past        6032           7.5  127585  3.366412
# 587                                           The Abyss         808           7.1    2756  3.267406
# 341   Percy Jackson & the Olympians: The Lightning T...        2010           6.0   32657  3.234864
# 1542                                              Speed        1783           6.8    1637  3.228102
# 261                               Live Free or Die Hard        2089           6.4    1571  3.227411
# 989                                      Baby's Day Out         274           5.8   11212  3.194140
# 129                                                Thor        6525           6.6   10195  3.193403

# Plot graph for Hybrid Recommendation

attrs = {"super_title": 'Hybrid Recommendation for different user',
        "ax1_movie_title": hybrid_recom_user_1['title'],
        "ax1_data":  hybrid_recom_user_1['est'],
        "ax1_color": 'cornflowerblue',
        "ax1_annotate_label": '{:,.3f}',
        "ax1_title": "Hybrid Recommendation for user1\n \n Finding movies base on 'Avatar' movie information",
        "ax1_xlabel": "SVD prediction rating",
        "ax1_set_scale": False,
        "ax1_xaxis_formatter": '{x:.2f}',
        "ax1_xlim": (2.95,3.45),

        "ax2_movie_title": hybrid_recom_user_500['title'],
        "ax2_data":  hybrid_recom_user_500['est'],
        "ax2_color": 'hotpink',
        "ax2_annotate_label": '{:,.3f}',
        "ax2_title": "Hybrid Recommendation for user500 \n \n Finding movies base on 'Avatar' movie information",
        "ax2_xlabel": "SVD prediction rating",
        "ax2_set_scale": False,
        "ax2_xaxis_formatter": '{x:.3f}',
        "ax2_xlim": (3,3.6)
        }

plot_graph(attrs)