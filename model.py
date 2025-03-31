import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Union
from openai import OpenAI
import time
import os

class SpotifyRecommender:
    def __init__(self, data_path: str = "spotifydata.csv", api_key: str = None):
        # Initialize OpenAI client
        if not api_key:
            raise ValueError("API key must be provided")
        self.openai_client = OpenAI(api_key=api_key)
        
        # Define features and their expected dtypes
        self.features = [
            "danceability", "energy", "key", "loudness", "mode",
            "speechiness", "acousticness", "instrumentalness",
            "liveness", "valence", "tempo", "duration_ms"
        ]
        self.feature_dtypes = {
            "danceability": "float64",
            "energy": "float64",
            "key": "int64",
            "loudness": "float64",
            "mode": "int64",
            "speechiness": "float64",
            "acousticness": "float64",
            "instrumentalness": "float64",
            "liveness": "float64",
            "valence": "float64",
            "tempo": "float64",
            "duration_ms": "float64"
        }
        
        self.df = self._load_and_prepare_data(data_path)
        self.similarity_matrix = self._compute_similarity_matrix()

    def _load_and_prepare_data(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess the Spotify dataset"""
        # Load data with specified dtypes
        df = pd.read_csv(data_path)
        
        # Select and copy relevant columns
        columns_to_keep = ["track_id", "track_name", "track_artist"] + list(self.features)
        df_filtered = df[columns_to_keep].copy()
        
        # Remove duplicates and missing values
        df_filtered = df_filtered.drop_duplicates(subset="track_name").copy()
        df_clean = df_filtered.dropna().copy()
        
        # Convert features to correct dtypes before scaling
        for feature, dtype in self.feature_dtypes.items():
            df_clean[feature] = df_clean[feature].astype(dtype)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_clean[self.features])
        
        # Create new DataFrame for scaled features to avoid dtype issues
        df_scaled = pd.DataFrame(scaled_features, columns=self.features, index=df_clean.index)
        
        # Combine with non-feature columns
        df_clean = pd.concat([
            df_clean[["track_id", "track_name", "track_artist"]],
            df_scaled
        ], axis=1)
        
        return df_clean.reset_index(drop=True)

    def _compute_similarity_matrix(self) -> List[List[float]]:
        """Compute cosine similarity matrix for all tracks"""
        return cosine_similarity(self.df[self.features])

    def _get_ai_explanation(self, seed_track: str, recommendations: List[Dict]) -> str:
        """Get AI-generated explanation using chat completions API"""
        try:
            # Prepare the prompt
            rec_list = "\n".join([f"- {r['track_name']} by {r['artist']} (similarity: {r['similarity_score']:.2f})" 
                       for r in recommendations])
            features_list = ", ".join(self.features)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a music expert that explains why songs are similar based on their audio features."
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Explain why these songs are musically similar to '{seed_track}'.
                        Consider these audio features: {features_list}
                        
                        Recommended tracks with similarity scores:
                        {rec_list}
                        
                        Provide a concise 2-3 sentence explanation focusing on musical characteristics and why each recommended song {rec_list} is similar to the {seed_track}.
                        Talk about each song, in keypoints/numbers.
                        """
                    }
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating explanation: {str(e)}")
            return f"Could not generate explanation: {str(e)}"

    def get_all_tracks(self) -> List[Dict[str, str]]:
        """Get list of all available tracks for API"""
        return self.df[["track_name", "track_artist"]].to_dict(orient="records")

    def recommend_tracks(self, track_name: str, n: int = 5, include_explanation: bool = False) -> Dict[str, Union[List[Dict], str]]:
        """
        Get recommendations with optional AI explanation
        
        Returns:
        {
            "seed_track": str,
            "recommendations": List[Dict],
            "explanation": str,
            "error": str
        }
        """
        if track_name not in self.df["track_name"].values:
            return {
                "seed_track": track_name,
                "recommendations": [],
                "explanation": "",
                "error": f"Track '{track_name}' not found"
            }
        
        idx = self.df.index[self.df["track_name"] == track_name][0]
        similarity_scores = list(enumerate(self.similarity_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for i in similarity_scores[1:n+1]:
            recommendations.append({
                "track_name": self.df.iloc[i[0]]["track_name"],
                "artist": self.df.iloc[i[0]]["track_artist"],
                "similarity_score": float(i[1])
            })
        
        # Get AI explanation if requested
        explanation = ""
        if include_explanation and recommendations:
            explanation = self._get_ai_explanation(track_name, recommendations)
        
        return {
            "seed_track": track_name,
            "recommendations": recommendations,
            "explanation": explanation,
            "error": ""
        }