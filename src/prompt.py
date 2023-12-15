import os
import pandas as pd
from openai import OpenAI
from func import *
from sklearn.model_selection import train_test_split
import re

class RecommendPrompt():
    def __init__(self, data):
        self.client = OpenAI()
        self.data = data
        self.train_data, self.test_data = train_test_split(data, test_size=0.2, random_state=42)
        self.system_prompt = "You are a movie recommender system that will compare user previous watch history and ratings."
        
    def prompt_recommend_with_user_history(self, user_id):
        if user_id in self.data['userId'].values:
            train_data_user = self.train_data[self.train_data["userId"] == user_id]
            titles_list, ratings_list = train_data_user["title"], train_data_user["rating"]
            messages=[
                {"role": "system", "content": f"{self.system_prompt} Based on the genre and ratings of those movies that this particular user gave, please recommend 10 movie that this user would enjoy watching and predict the rating of these 10 movies given by this user"},
                {"role": "user", "content": f"User {user_id}'s previously watch movies include {titles_list} and the corresponded ratings are {ratings_list}. Please suggest 10 movie based on the watch history and predict ratings of those 10 movies based on user preferences. Format: [n. <Movie Name> (<Year>) - Predicted Rating: <Rating>]"}
            ]
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=messages
            )
            response = completion.choices[0].message.content
            recommendations = self.extract_string1(response)
            return messages, response, recommendations
        else:
            print(f"{user_id} is not a valid user ID, please try again")
            return None, None, None
            
    def prompt_recommend_with_partner_history(self, userA, userB):
        if userA and userB in self.data['userId'].values:
            train_data_userA = self.train_data[self.train_data["userId"] == userA]
            titles_list_A, ratings_list_A = train_data_userA["title"], train_data_userA["rating"]
            train_data_userB = self.train_data[self.train_data["userId"] == userB]
            titles_list_B, ratings_list_B = train_data_userB["title"], train_data_userB["rating"]
            messages=[
                {"role": "system", "content": f"{self.system_prompt} Based on the genre and ratings of movies from 2 users, please recommend 10 movie that these 2 users would enjoy watching together and predict the rating of these 10 movies"},
                {"role": "user", "content": f"User {userA}'s previously watch movies include {titles_list_A} and the corresponded ratings are {ratings_list_A}. User {userB}'s previously watch movies include {titles_list_B} and the corresponded ratings are {ratings_list_B} Please suggest 10 movie that both users would enjoy watching based on the watch histories and predict specific ratings of those 10 movies based on user preferences. Format: [n. <Movie Name> (<Year>) - Predicted Rating: <Rating>]"}
            ]
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=messages
            )
            response = completion.choices[0].message.content
            recommendations = self.extract_string1(response)
            return messages, response, recommendations
        else:
            print(f"{userA} or {userB} is not a valid user ID, please try again")
            return None, None, None
        
    def prompt_with_genre(self, user_id):
        if user_id in self.data['userId'].values:
            user_data = self.train_data[self.train_data["userId"] == user_id]
            user_data['genres'] = user_data['genres'].str.split('|')
            genres_expanded = user_data.explode('genres')
            genre_ratings = genres_expanded.groupby('genres')['rating'].mean()
            top_3_genres = genre_ratings.sort_values(ascending=False).head(3).index.tolist()
            messages=[
                {"role": "system", "content": "You are a movie recommender system that will recommend 10 movie in this user's favorite genres and predict the rating of these 10 movies given by this user"},
                {"role": "user", "content": f"User {user_id}'s favorite genre include {top_3_genres}. Please suggest 10 movie in these genre for the user. Format: [n. <Movie Name> (<Year>) - Predicted Rating: <Rating>]"}
            ]
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=messages
            )
            response = completion.choices[0].message.content
            recommendations = self.extract_string1(response)
            return messages, response, recommendations
        else:
            print(f"{userId} is not a valid user ID, please try again")
            return None, None, None

    def prompt_with_similar_user(self, user_id):
        if user_id in self.data['userId'].values:
            most_similar = get_similar_users(user_id, self.train_data)
            train_data_user = self.train_data[self.train_data["userId"] == most_similar]
            titles_list, ratings_list = train_data_user["title"], train_data_user["rating"]
            messages=[
                {"role": "system", "content": "You are a movie recommender system that will suggest movies that the user may also like based on similar user's watch histories. From the most similar user's watch histories, please recommend 10 movies that this user would enjoy watching and predict the rating of these 10 movies given by this user"},
                {"role": "user", "content": f"User {user_id}'s most similar user is user {most_similar}, which has previously watched {titles_list} and the corresponded ratings are {ratings_list}. Please suggest 10 movies that user {user_id} may also like and provide rating prediction of those 10 movies based on user preferences. Format: [n. <Movie Name> (<Year>) - Predicted Rating: <Rating>]"}
            ]
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=messages
            )
            response = completion.choices[0].message.content
            recommendations = self.extract_string1(response)
            return messages, response, recommendations
        else:
            print(f"{userId} is not a valid user ID, please try again")
            return None, None, None

    def prompt_for_rating(self, user_id):
        if user_id in self.data['userId'].values:
            train_data_user = self.train_data[self.train_data["userId"] == user_id]
            train_titles_list, train_ratings_list = train_data_user["title"], train_data_user["rating"]
            test_data_user = self.test_data[self.test_data["userId"] == user_id]
            test_title_list, true_ratings = test_data_user["title"], train_data_user["rating"]
            messages=[
                {"role": "system", "content": "You are a movie rating prediction system that will predict rating with a list of titles given"},
                {"role": "user", "content": f"User {user_id}'s previously watch movies include {train_titles_list} and the corresponded ratings are {train_ratings_list}. Based on these ratings, please provide a list of prediction rating corresponded to {test_title_list}. Format: [n. <Movie Name> (<Year>) - Predicted Rating: <Rating>]"}
            ]
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=messages
            )
            response = completion.choices[0].message.content
            recommendations = self.extract_string1(response)
            rms = self.evaluation(recommendations, user_id)
            print(f"Recommendation Rating RMSE: {rms}")
            return messages, response, recommendations
        else:
            print(f"{userId} is not a valid user ID, please try again")
            return None, None, None
    
    def extract_string1(self, string):
        lines = string.split('\n')
        pattern1 = r'(\d+)\. (.*?) \((\d+)\)'
        pattern2 = r'(\d+\.\d+)'
        movie_info = []
        for line in lines:
            if '-' in line:
                match1 = re.search(pattern1, line)
                match2 = re.search(pattern2, line)
                if match1 and match2:
                    rank = match1.group(1)
                    title = match1.group(2)
                    year = match1.group(3)
                    rating = match2.group(1)
                    movie_info.append((title, rating))
        return movie_info
    
    def evaluation(self, recommendation, user_id):
        from sklearn.metrics import mean_squared_error
        y_pred = []
        for movie, rating in recommendation:
            y_pred.append(float(rating))
        y_true = self.test_data[self.test_data["userId"] == user_id]["rating"].tolist()
        rms = mean_squared_error(y_true, y_pred, squared=True)
        return rms
    
    def accuracy(self, recommendation, user_id):
        test_data_user = self.test_data[self.test_data["userId"] == user_id]
        test_title, test_ratings = test_data_user["title"], test_data_user["rating"]
        correct = 0
        test_title_list = list(test_title)
        for title, rating in recommendation:
            if "The" in title:
                title = title[4:]
            find_candidate = 0
            for movie in test_title_list:
                if title in movie:
                    print(f"{title} in test")
                    correct += 1
                    break
        return correct / len(recommendation)
        
        