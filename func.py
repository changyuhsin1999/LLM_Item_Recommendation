def get_user_watch_history(user_id, data, fraction=0.8):
    """
    Extracts a fraction of a specified user's watch history from a given dataset.
    
    Args:
    user_id (int): The ID of the user whose watch history is to be extracted.
    data (DataFrame): The dataset containing user watch histories.
    fraction (float, optional): The fraction of the user's watch history to be sampled. Defaults to 0.8.

    Returns:
    titles_list: A list containing all the movie titles in the user's sampled watch history.
    ratings_list: A list containing all the corresponding ratings of the movies in the user's sampled watch history.
    """
    user_watch_history = data[data['userId'] == user_id]
    user_history_train = user_watch_history.sample(frac=fraction, random_state=42)
    
    # Extracting titles and ratings as separate lists
    titles_list = user_history_train['title'].tolist()
    ratings_list = user_history_train['rating'].tolist()

    return titles_list, ratings_list


def find_top_3_genre(user_id,data):
    """
    Identifies the top three genres based on average ratings from a specific user's watch history.

    This function processes the user's watch history to calculate the average rating for each genre. It then identifies and returns the top three genres based on these average ratings.

    Args:
    user_id (int): The ID of the user whose top genres are to be determined.
    data (DataFrame): The dataset containing user watch histories along with genres and ratings.

    Returns:
    top_3_genres: A list of the top three genres with the highest average ratings in the user's watch history.
    
    """
    user_data = data[data['userId'] == user_id].reset_index(drop=True)

    # Genres are combined in a single column, split them and explode into separate rows
    user_data['genres'] = user_data['genres'].str.split('|')
    genres_expanded = user_data.explode('genres')

    # Group by genre and calculate average rating for each genre
    genre_ratings = genres_expanded.groupby('genres')['rating'].mean()

    # Get the genre with the highest average rating
    top_3_genres = genre_ratings.sort_values(ascending=False).head(3).index.tolist()

    return top_3_genres