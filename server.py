from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import faiss
import pandas as pd

app = FastAPI(title="Places Recommendation API")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

places = pd.read_csv("Recommendation Notebooks/Datasets/places.csv")
users_dataset = pd.read_csv("Recommendation Notebooks/Datasets/Users.csv")
Places_embeddings_matrix = np.load("Recommendation Notebooks/VectorDatabase/Places_Embeddings.npy")
index = faiss.read_index("Recommendation Notebooks/VectorDatabase/faiss_Places_Embeddings.index")


def get_similar_places(target_id: int, k: int = 6):
    row_idx = target_id - 1
    query_vector = Places_embeddings_matrix[row_idx: row_idx + 1]

    distance, indices = index.search(query_vector, k)
    distance = distance.flatten()
    indices = indices.flatten()

    for dist, idx in zip(distance, indices):
        if idx == row_idx:
            continue
        row = places.iloc[idx, :].copy()
        row["distance"] = float(dist)
        yield row


@app.get("/")
def root():
    return {"message": "Places Recommendation API is running"}


@app.get("/similar_places/{place_id}")
def similar_places(place_id: int, k: int = 10):
    if place_id < 1 or place_id > len(places):
        raise HTTPException(status_code=404, detail=f"Place ID {place_id} not found")

    results = pd.DataFrame(get_similar_places(target_id=place_id, k=k + 1))

    if results.empty:
        raise HTTPException(status_code=404, detail="No similar places found")

    results = results.sort_values(by="distance", ascending=False)
    return results.to_dict(orient="records")

@app.get("/content/{user_id}")
def Get_Similar_places_Content(user_id:str):
    top_Recommendation = 10
    fav_places = users_dataset[(users_dataset["user_id"] == user_id) & (users_dataset["rating"] > 3)]["place_id"].values
    
    if len(fav_places) == 0:
        print(f"User {user_id} has no highly rated places yet.")
        return pd.DataFrame() 
        
    fav_vectors = Places_embeddings_matrix[fav_places - 1]
    
    user_profile_vector = np.mean(fav_vectors, axis=0)
    
    user_profile_vector = user_profile_vector.reshape(1, -1).astype(np.float32)
    
    faiss.normalize_L2(user_profile_vector)
    
    k = len(fav_places) + top_Recommendation
    distances, indices = index.search(user_profile_vector, k)
    
    distances = distances.flatten()
    indices = indices.flatten()
    
    recommend = []
    for dist, idx in zip(distances, indices):
        place_id = places.iloc[idx]["ID"]
        
        if place_id not in fav_places:
            row = places.iloc[idx].copy()
            row["similarity_score"] = dist
            recommend.append(row)
            
        if len(recommend) >= top_Recommendation:
            return pd.DataFrame(recommend).to_dict(orient="records") 