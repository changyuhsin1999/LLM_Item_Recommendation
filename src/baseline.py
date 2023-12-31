import os
import pandas as pd
from openai import OpenAI
from func import *

class MostPopularRecommendor():
    def __init__(self, train_data):
        client = OpenAI()
        self.train_data = train_data
    
    def pred(self):
        movie_popularity = self.train_data.groupby('title').size().sort_values(ascending=False)
        baseline_pred = []
        for movie in movie_popularity.index:
            baseline_pred.append(movie)
            if len(baseline_pred) >= 10:
                break
        return baseline_pred
    
    def accuracy(self, movie_pred, test_title):
        correct = 0
        test_title_list = list(test_title)
        for movie in movie_pred:
            if movie in test_title_list:
                # print(f"{movie} in test")
                correct += 1
                break
        return correct / len(movie_pred)
        
        

class RecommendPrompt():
    def __init__(self, data):
        client = OpenAI()
        self.data = data
        # self.training_data = 
        
    def prompt_recommend_with_user_history(self, userId):
        if user_id in self.data['userId'].values:
            titles_list, ratings_list = get_user_watch_history(user_id, self.data)
            messages=[
                {"role": "system", "content": f"{system_prompt} Based on the genre and ratings of those movies that this particular user gave, please recommend 10 movie that this user would enjoy watching and predict the rating of these 10 movies given by this user"},
                {"role": "user", "content": f"User {user_id}'s previously watch movies include {titles_list} and the corresponded ratings are {ratings_list}. Please suggest 10 movie based on the watch history and predict ratings of those 10 movies based on user preferences"}
            ]
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            reponse = completion.choices[0].message.content
            return messages, response
        else:
            print(f"{userId} is not a valid user ID, please try again")
            return None, None
            
    def prompt_recommend_with_partner_history(self, userA, userB):
        if user_A and user_B in self.data['userId'].values:
            titles_list_A, ratings_list_A = get_user_watch_history(user_A, self.data)
            titles_list_B, ratings_list_B = get_user_watch_history(user_B, self.data)
            messages=[
                {"role": "system", "content": f"{system_prompt} Based on the genre and ratings of movies from 2 users, please recommend 10 movie that these 2 users would enjoy watching together and predict the rating of these 10 movies"},
                {"role": "user", "content": f"User {user_A}'s previously watch movies include {titles_list_A} and the corresponded ratings are {ratings_list_A}. User {user_B}'s previously watch movies include {titles_list_B} and the corresponded ratings are {ratings_list_B} Please suggest 10 movie that both users would enjoy watching based on the watch histories and predict specific ratings of those 10 movies based on user preferences"}
            ]
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            reponse = completion.choices[0].message.content
            return messages, response
        else:
            print(f"{userA} or {userB} is not a valid user ID, please try again")
            return None, None
        
    def prompt_with_genre(self, userId):
        if user_id in self.data['userId'].values:
            top_3_genre = find_top_3_genre(user_id_genre,self.data)
            messages=[
                {"role": "system", "content": "You are a movie recommender system that will recommend 10 movie in this user's favorite genres and predict the rating of these 10 movies given by this user"},
                {"role": "user", "content": f"User {user_id_genre}'s favorite genre include {top_3_genre}. Please suggest 10 movie in these genre for the user"}
            ]
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            reponse = completion.choices[0].message.content
            return messages, response
        else:
            print(f"{userId} is not a valid user ID, please try again")
            return None, None

    def prompt_with_similar_user(self, userId):
        if user_id in self.data['userId'].values:
            most_similar = get_similar_users(user_id_sim, merged_df)
            titles_list, ratings_list = get_user_watch_history(most_similar, merged_df)
            messages=[
                {"role": "system", "content": "You are a movie recommender system that will suggest movies that the user may also like based on similar user's watch histories. From the most similar user's watch histories, please recommend 10 movies that this user would enjoy watching and predict the rating of these 10 movies given by this user"},
                {"role": "user", "content": f"User {user_id_sim}'s most similar user is user {most_similar}, which has previously watched {titles_list} and the corresponded ratings are {ratings_list}. Please suggest 10 movies that user {user_id_sim} may also like and provide rating prediction of those 10 movies based on user preferences"}
            ]
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            reponse = completion.choices[0].message.content
            return messages, response
        else:
            print(f"{userId} is not a valid user ID, please try again")
            return None, None

    def prompt_for_rating(self, userId):
        if user_id in self.data['userId'].values:
            titles_list, ratings_list = get_user_watch_history(user_id_rating, merged_df)
            test_title_list, true_ratings = get_user_watch_history_20(user_id_rating, merged_df, fraction=0.8)
            messages=[
                {"role": "system", "content": "You are a movie rating prediction system that will predict rating with a list of titles given"},
                {"role": "user", "content": f"User {user_id_rating}'s previously watch movies include {titles_list} and the corresponded ratings are {ratings_list}. Based on these ratings, please provide a list of prediction rating corresponded to {test_title_list} with the format of movie title - Predicted Rating: ratings"}
            ]
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            reponse = completion.choices[0].message.content
            return messages, response
        else:
            print(f"{userId} is not a valid user ID, please try again")
            return None, None
        
    