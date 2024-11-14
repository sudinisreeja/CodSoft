import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading the data from the csv file to a pandas dataframe
movies_data = pd.read_csv('/content/movies.csv')

# Printing the first 5 rows of the dataframe
movies_data.head()

# Selecting the relevant features for recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Check if the columns exist in the dataframe
# Print a warning message if a column is not found
for feature in selected_features:
    if feature not in movies_data.columns:
        print(f"Warning: Column '{feature}' not found in the dataframe.")
    else:
        # Replacing the null values with an empty string only if the column exists
        movies_data[feature] = movies_data[feature].fillna('')

# Combining all the selected features
# Use only the features that exist in the dataframe
existing_features = [f for f in selected_features if f in movies_data.columns]
combined_features = movies_data[existing_features[0]].astype(str)  # Initialize with the first feature
for feature in existing_features[1:]:
    combined_features = combined_features + ' ' + movies_data[feature].astype(str)

# Converting the text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)

# Getting the movie name from the user
movie_name = input('Enter your favorite movie name: ')

# Creating a list with all the movie names given in the dataset
list_of_all_titles = movies_data['title'].tolist()

# Finding the closest match for the movie name given by the user
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

if not find_close_match:
    print("Sorry, no matching movie found!")
else:
    close_match = find_close_match[0]

    # Finding the index of the movie with the title
    index_of_the_movie = movies_data[movies_data.title == close_match].index[0]

    # Getting a list of similar movies
    similarity_score = list(enumerate(similarity[index_of_the_movie]))

    # Sorting the movies based on their similarity score
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Print the name of similar movies based on the index
    print('Movies suggested for you:\n')

    i = 1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data.loc[index, 'title']
        if i < 30:
            print(i, '.', title_from_index)
            i += 1