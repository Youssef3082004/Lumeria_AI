from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random
from database import get_live_data
from recommender import *
import numpy as np 
import faiss

app = FastAPI(title="Lumeria AI API")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

users_dataset , places = get_live_data()
Places_embeddings_matrix = np.load("VectorDatabase/Places_Embeddings.npy")
index = faiss.read_index("VectorDatabase/faiss_Places_Embeddings.index")

@app.get("/")
def root():
    return {"status": "Lumeria Recommendation Engine is Online"}

@app.get("/recommend/popular")
def popular_api(location: str = None):
    return get_popular_list(users_dataset, places, location)

@app.get("/recommend/user/{user_id}")
def user_api(user_id: str):
    # 2. محاولة جلب توصيات بناءً على ذوق اليوزر (FAISS)
    recommendations = get_user_based_faiss(user_id, users_dataset, places)
    
    # 3. لو اليوزر جديد (النتائج فاضية) ->  أماكن عشوائية
    if not recommendations:
        # تحويل جدول الأماكن لقائمة من القواميس
        all_places_list = places.to_dict(orient="records")
        num_to_select = min(10, len(all_places_list))
        
        # اختيار الأماكن العشوائية
        random_recommendations = random.sample(all_places_list, k=num_to_select)
        
        return random_recommendations

    return recommendations


@app.get("/recommend/similarplaces/{place_id}")
def similar_places(place_id:int, k:int = 10):
    results = pd.DataFrame(get_similar_places(Places_embeddings_matrix=Places_embeddings_matrix,index=index,places=places,target_id=place_id, k=k + 1)).sort_values(by="distance", ascending=False)
    return results.to_dict(orient="records")


@app.get("/recommend/content/{user_id}")
def Get_Similar_places_Content(user_id:str, k:int = 10):
    top_Recommendation = 10
    fav_places = users_dataset[(users_dataset["user_id"] == user_id) & (users_dataset["rating"] > 3)]["place_id"].values
    k = len(fav_places) + top_Recommendation

    
    if len(fav_places) == 0:
        return random.sample(places.to_dict(orient="records"),k=k) 
        
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
            row["similarity_score"] = dist
            recommend.append(row)
            
        if len(recommend) >= top_Recommendation:
            return pd.DataFrame(recommend).to_dict(orient="records") 
