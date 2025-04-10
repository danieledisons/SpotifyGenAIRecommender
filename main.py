from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model import SpotifyRecommender
from typing import List, Dict
import uvicorn
from dotenv import load_dotenv
import os

app = FastAPI(title="Spotify Track Recommender API")
load_dotenv()

# # Initialize recommender (will load data on startup)
# recommender = SpotifyRecommender(api_key=os.getenv("OPENAI_API_KEY"))
recommender = None

@app.on_event("startup")
def load_recommender():
    global recommender
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    recommender = SpotifyRecommender(api_key=api_key)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root():
    return {"message": "Welcome to the Spotify Recommendation API"}

@app.get("/tracks", response_model=List[Dict[str, str]])
def get_all_tracks():
    """Get list of all available tracks"""
    return recommender.get_all_tracks()

@app.get("/recommend/{track_name}")
def get_recommendations(track_name: str, n: int = 5, explain: bool = False):
    """Get track recommendations with optional AI explanation"""
    result = recommender.recommend_tracks(track_name, n, include_explanation=explain)
    
    if result["error"]:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return {
        "seed_track": track_name,
        "recommendations": result["recommendations"],
        "explanation": result["explanation"] if explain else None
    }

@app.get("/search")
def search_tracks(query: str, limit: int = 10):
    """Search tracks by name"""
    all_tracks = recommender.get_all_tracks()
    matches = [
        track for track in all_tracks
        if query.lower() in track["track_name"].lower()
    ][:limit]
    return {"results": matches}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

