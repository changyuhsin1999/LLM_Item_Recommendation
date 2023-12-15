import os
import pandas as pd
from openai import OpenAI
from func import get_user_watch_history
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

class ThreeStepRecommender:
    def __init__(self, user_id, train_data, candidate_size = 100):
        self.client = OpenAI()
        self.user_id = user_id
        self.candidate_size = candidate_size
        self.train_data = train_data
        self.train_user_df = self.train_data[self.train_data["userId"] == self.user_id]
        self.train_title, self.train_rating = self.train_user_df["title"], self.train_user_df["rating"]

    def filter_user(self):

        def get_similar_users(user_id, data):
            user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')
            user_movie_matrix = user_movie_matrix.fillna(0)
            similarity_matrix = cosine_similarity(user_movie_matrix)
            similarity_df = pd.DataFrame(similarity_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)
            similar_users = similarity_df[self.user_id].sort_values(ascending=False)
            return similar_users

        similar_users = list(get_similar_users(self.user_id, self.train_data).iloc[:self.candidate_size].index)
        train_similar_df = self.train_data[self.train_data["userId"].isin(similar_users)]
        movie_popularity = train_similar_df.groupby('title').size().sort_values(ascending=False)
        self.candidate1 = list(movie_popularity.head(self.candidate_size).head(self.candidate_size).index)

    def filter_movie(self):
        train_movie = self.train_user_df["movieId"]
        watched = pd.unique(train_movie).tolist()
        
        user_item_matrix = self.train_data.pivot_table(index='userId', columns='movieId', values='rating')
        user_item_matrix = user_item_matrix.fillna(0)
        item_similarity = cosine_similarity(user_item_matrix.T)
        
        def find_similar_movies(target_item_id, data):
            target_item_index = user_item_matrix.columns.get_loc(target_item_id)
            similarities = item_similarity[target_item_index]
            similar_items_df = pd.DataFrame({'movieId': user_item_matrix.columns, 'similarity_score': similarities})
            similar_items_df = similar_items_df.sort_values(by='similarity_score', ascending=False)
            N = self.candidate_size
            top_similar_items = similar_items_df.head(N)
            return top_similar_items

        similar_movies = []
        for movie in watched:
            similar_movies.append(find_similar_movies(movie, self.train_data))
        similar_df = pd.concat(similar_movies)

        movie_popularity = similar_df.groupby('movieId').size().sort_values(ascending=False)
        self.candidate2 = list(movie_popularity.head(self.candidate_size).head(self.candidate_size).index)
        self.candidate2 = self.train_data.loc[self.train_data['movieId'].isin(self.candidate2), 'title'].tolist()
        self.candidate = list(set(self.candidate1) | set(self.candidate2))

    def step1(self):
        self.movie_rating = ""
        for i in range(len(self.train_title)):
            self.movie_rating += f"{self.train_title.iloc[i]}: {self.train_rating.iloc[i]} \n"

        if self.user_id in self.train_data['userId'].values:
            titles_list, ratings_list = get_user_watch_history(self.user_id, self.train_data)
            messages=[
                # {"role": "user", "content": f"self.candidate Set(self.candidate movies): "},
                {"role": "user", "content": f"The movies I have watched(watched movies): {self.movie_rating}"},
                {"role": "user", "content": f"Step 1: What features are most important to me when selecting movies? "},
            ]
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=messages
            )

        self.answer1 = (completion.choices[0].message.content)


    def step2(self):

        messages=[
            {"role": "user", "content": f"The movies I have watched(watched movies) and their ratings: {self.movie_rating}"},
            {"role": "user", "content": f"Step 1: What features are most important to me when selecting movies? "},
        ]

        messages.append({"role": "assistant", "content": self.answer1})
                        
        step2 = "You will select the movies (at most 10 movies) that appeal to me the most from the list of movies \
            I have watched, based on my personal preferences. The selected movies will be presented in descending \
            order of preference. (Format: no. a watched movie)."
            
        messages.append({"role": "user", "content": step2})
        
        self.messages = messages
                        
        completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=messages
            )

        self.answer2 = (completion.choices[0].message.content)
        
    def step3(self):

        self.messages.append({"role": "assistant", "content": self.answer2})

        self.messages.append({"role": "user", "content": f"self.candidate Set(self.candidate movies): {', '.join(self.candidate)}"},)
                        
        step3 = "Can you recommend 10 different movies only from the self.candidate Set similar to the selected \
            movies I've watched (Format: [<n>. <a watched movie> : <a self.candidate movie>])?" + f"self.candidate Set(self.candidate movies): {', '.join(self.candidate)}"
            
        self.messages.append({"role": "user", "content": step3})
                        
        completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=self.messages
            )

        self.answer3 = (completion.choices[0].message.content)




        def parse_answer3(answer3):
            lines = self.answer3.split('\n')
            pattern = r': (.*?) \((\d+)\)'
            movie_pred = []
            for line in lines:
                match = re.search(pattern, line)
                if match:
                    title = match.group(1)
                    year = match.group(2)
                    movie_pred.append((title, year))
            return movie_pred

        self.movie_pred = parse_answer3(self.answer3)
        
    def get_pred(self):
        self.filter_user()
        self.filter_movie()
        self.step1()
        print("finish step 1")
        self.step2()
        print("finish step 2")
        self.step3()
        print("finish step 3")
        return self.movie_pred
        

    def accuracy(self, movie_pred, test_title):
        correct = 0
        test_title_list = list(test_title)
        for title, year in movie_pred:
            if "The" in title:
                title = title[4:]
            find_candidate = 0
            for movie in self.candidate:
                if title in movie:
                    find_candidate = 1
                    break
            if not find_candidate:
                pass
                # print(f"{title} not in self.candidate")
            for movie in test_title_list:
                if title in movie:
                    # print(f"{title} in test")
                    correct += 1
                    break
        return correct / len(movie_pred)
    
    
