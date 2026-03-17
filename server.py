from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import faiss
import pandas as pd

app = FastAPI(title="Places Recommendation API")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

places = pd.read_csv("Datasets/places.csv")
Places_embeddings_matrix = np.load("Places_Embeddings.npy")
index = faiss.read_index("faiss_Places_Embeddings.index")


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