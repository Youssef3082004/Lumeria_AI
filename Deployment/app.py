from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random
from database import get_live_data
from recommender import *
import numpy as np 
import faiss
import pandas as pd

app = FastAPI(title="Lumeria AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

Places_embeddings_matrix = np.load("VectorDatabase/Places_Embeddings.npy")
index = faiss.read_index("VectorDatabase/faiss_Places_Embeddings.index")

@app.get("/")
def root():
    return {"status": "Lumeria Recommendation Engine is Online"}

@app.get("/recommend/popular")
def popular_api(location: str = None):
    users_dataset, places = get_live_data()
    return get_popular_list(users_dataset, places, location)

@app.get("/recommend/user/{user_id}")
def user_api(user_id: str):
    users_dataset, places = get_live_data()
    
    recommendations = get_user_based_faiss(user_id, users_dataset, places)
    
    all_places_list = places.to_dict(orient="records")
    target_count = 10 

    if not recommendations:
        num_to_select = min(target_count, len(all_places_list))
        return random.sample(all_places_list, k=num_to_select)

    if len(recommendations) < target_count:
        existing_ids = [p['ID'] for p in recommendations]
        remaining_places = [p for p in all_places_list if p['ID'] not in existing_ids]
        needed_count = target_count - len(recommendations)
        extra_random_picks = random.sample(remaining_places, k=min(needed_count, len(remaining_places)))
        full_recommendations = recommendations + extra_random_picks
        return full_recommendations

    return recommendations[:target_count]

@app.get("/recommend/similarplaces/{place_id}")
def similar_places(place_id:int, k:int = 10):
    users_dataset, places = get_live_data()
    
    results = pd.DataFrame(get_similar_places(
        Places_embeddings_matrix=Places_embeddings_matrix,
        index=index,
        places=places,
        target_id=place_id, 
        k=k + 1
    )).sort_values(by="distance", ascending=False)
    
    return results.to_dict(orient="records")

@app.get("/recommend/content/{user_id}")
def Get_Similar_places_Content(user_id:str, k:int = 10):
    users_dataset, places = get_live_data()
    
    top_Recommendation = 10
    fav_places = users_dataset[(users_dataset["user_id"] == user_id) & (users_dataset["rating"] > 3)]["place_id"].values
    k = len(fav_places) + top_Recommendation
    
    if len(fav_places) == 0:
        return random.sample(places.to_dict(orient="records"), k=min(k, len(places)))
    
    fav_vectors = Places_embeddings_matrix[fav_places - 1]
    user_profile_vector = np.mean(fav_vectors, axis=0).reshape(1, -1).astype(np.float32)
    
    faiss.normalize_L2(user_profile_vector)
    distances, indices = index.search(user_profile_vector, k)
    distances = distances.flatten()
    indices = indices.flatten()
    
    recommend = []
    for dist, idx in zip(distances, indices):
        place_id = places.iloc[idx]["ID"]
        if place_id not in fav_places:
            row = places.iloc[idx].copy()
            row["similarity_score"] = float(dist)
            recommend.append(row)
            
        if len(recommend) >= top_Recommendation:
            break
            
    return pd.DataFrame(recommend).to_dict(orient="records")
