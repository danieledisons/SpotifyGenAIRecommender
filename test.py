import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import time
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Get the API key from the environment variable
api_key = os.getenv('OPENAI_API_KEY')

# Load the dataset
df = pd.read_csv("spotifydata.csv")
df.head()

# Select relevant features for content-based filtering
features = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", 
            "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]

# Drop non-numeric and irrelevant columns
df_filtered = df[["track_id", "track_name", "track_artist"] + features]

# Handle missing values
df_filtered.dropna(inplace=True)

# Normalize feature values
scaler = StandardScaler()
df_filtered[features] = scaler.fit_transform(df_filtered[features])

# Compute cosine similarity
similarity_matrix = cosine_similarity(df_filtered[features])

# Function to recommend tracks
def recommend_tracks(track_name, n=5):
    if track_name not in df_filtered["track_name"].values:
        return "Track not found in the dataset."
    
    idx = df_filtered[df_filtered["track_name"] == track_name].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get recommended tracks and also capture their similarity scores
    recommended = []
    for i in similarity_scores[1:n+1]:
        rec_track = df_filtered.iloc[i[0]]["track_name"]
        rec_track_artist = df_filtered.iloc[i[0]]["track_artist"]
        score = i[1]
        recommended.append((rec_track, score))
    return recommended

# Function to get AI-generated explanation for recommendations
def get_ai_explanation(input_track, recommended_tracks):
    # Build the report text
    report_text = f"For the track '{input_track}', the following tracks were recommended based on cosine similarity:\n\n"
    for track, score in recommended_tracks:
        report_text += f"- {track} (Similarity Score: {score:.3f})\n"
    report_text += "\nPlease explain why these songs were recommended based on the cosine similarity of features such as danceability, energy, tempo, and others. Take about each of the songs that were recommended in a report format and relate it to their similarity score. Approach it like your talking to users that use music streaming platforms like Spotify."
    
    # Step 1: Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Create an assistant
    assistant = client.beta.assistants.create(
        name="SpotifyAiAssistant",
        instructions="Assist in creating and improving a recommendation system using cosine similarity to recommend tracks based on feature similarity, and provide explanations of the recommendations in an easily understandable way for users. Answer questions briefly, in a sentence or less.",
        model="gpt-4",
    )
    
    # Step 2: Create a thread for the conversation
    thread = client.beta.threads.create()

    # Step 3: Create a user message with the report text as input.
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=report_text,
    )
    
    # Step 4: Execute the run for the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    
    # Step 5: Wait for the run to complete
    time.sleep(5)
    
    # Step 6: Retrieve the messages added after the user message
    messages = client.beta.threads.messages.list(
        thread_id=thread.id,
        order="asc",  # Or "desc" if you want the newest first
        after=message.id
    )

    # Check if messages are returned
    if messages.data:
        message = messages.data[0]  # Access the first message in the list
        ai_reply = message.content[0].text.value
        return ai_reply
    else:
        return "No explanation available."

# Main logic
def main():
    input_track = input("Enter the track name for recommendations: ")

    # Get recommended tracks for the input track
    recommended_tracks = recommend_tracks(input_track)

    if isinstance(recommended_tracks, str):  # If the track is not found
        print(recommended_tracks)
    else:
        print("\nRecommended Tracks and Similarity Scores:")
        for track, score in recommended_tracks:
            print(f"- {track} (Similarity Score: {score:.3f})")

        # Get AI explanation for recommendations
        explanation = get_ai_explanation(input_track, recommended_tracks)
        print("\nAI Explanation Report:")
        print(explanation)

if __name__ == "__main__":
    main()
