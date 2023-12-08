# LLM_Item_Recommendation (AIPI 531 Final Project)
Team member: Cindy Chang, Yiyang Shao

Prompt Engineering of LLM for item recommendation task

Using Large Language Models (LLMs) as recommenders through strategic prompting involves crafting precise and contextually relevant queries to obtain tailored suggestions. In this project, we apply prompt engineering techniques to ask one of the most popular LLM (ChatGPT) to generate movie suggestions based on user's watch history and compare its result with a baseline collaborative filtering recommender system.

# Data
[MovieLens data](https://grouplens.org/datasets/movielens/)

We obtained data from the famous movie recommending dataset : MovieLens latest dataset with 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users.

# Data sourcing process and baseline recommender
Find out top 10 movies that has the highest rating with over 50 counts of ratings (most popular)

This can be our baseline recommender
![Screenshot](https://github.com/changyuhsin1999/LLM_Item_Recommendation/blob/main/images/Screenshot%202023-12-07%20at%203.26.53%20PM.png)

However, with the user watch histories and preferences, we can create a collaborative filtering recommender with user and item embeddings which can generate a more personalized movie suggestions for a specific user
![Screenshot](https://github.com/changyuhsin1999/LLM_Item_Recommendation/blob/main/images/Screenshot%202023-12-07%20at%203.45.29%20PM.png)

Or get a predicted rating from a specific user and movie set
![Screenshot](https://github.com/changyuhsin1999/LLM_Item_Recommendation/blob/main/images/Screenshot%202023-12-07%20at%203.53.37%20PM.png)
For more detail of the training process please visit [Collab filtering notebook](https://github.com/changyuhsin1999/LLM_Item_Recommendation/blob/main/Baseline_collab_filtering_movie_rec.ipynb) and run with Google Colab GPU

# Prompting with ChatGPT API
We included several zero-shot promptings using ChatGPT API:

1. Watch alone : ask ChatGPT to generate 10 movie suggestions and its predictive ratings based on given user watch history
2. Watch with partner: ask ChatGPT to generate 10 movie suggestions for a watch party given 2 different users's watch histories with different watch preferences and predict how would these 2 users will rate these 10 movie
3. Ask ChatGPT to generate 10 movie suggestions a user may also like based on most similar user's watch history

# How To Get Started
### Clone repo
### Prepare your environment

```
conda create --name openai_envir python=3.7.15
conda activate openai_envir
```
### Install requirement.txt

```
pip install -r requirements.txt
```
### Create an API key and a .env file
You can obtained an API key by signing up with ChatGPT and follow the step in OpenAI Quickstart page [OpenAI Quickstart](https://platform.openai.com/docs/quickstart?context=python)
create a .env file in your directory and add your API key into the file

```
OPENAI_API_KEY='your-api-key-here'
```
