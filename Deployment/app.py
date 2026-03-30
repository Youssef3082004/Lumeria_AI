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
    # 1. جلب التوصيات المبنية على ذوق المستخدم (الـ Collaborative)
    recommendations = get_user_based_faiss(user_id, users_dataset, places)
    
    # تحويل جدول الأماكن بالكامل لقائمة لاختيار العشوائي منها لاحقاً
    all_places_list = places.to_dict(orient="records")
    target_count = 10  # العدد المطلوب في الصفحة

    # الحالة الأولى: لو المستخدم جديد تماماً (لا توجد توصيات كولابريتيف)
    if not recommendations:
        num_to_select = min(target_count, len(all_places_list))
        return random.sample(all_places_list, k=num_to_select)

    # الحالة الثانية: يوجد توصيات كولابريتيف ولكن عددها أقل من 10
    if len(recommendations) < target_count:
        # استخراج المعرفات (IDs) الموجودة فعلياً في التوصيات الحالية لمنع التكرار
        existing_ids = [p['ID'] for p in recommendations]
        
        # تصفية كل الأماكن لاستبعاد ما تم اقتراحه بالفعل
        remaining_places = [p for p in all_places_list if p['ID'] not in existing_ids]
        
        # حساب كم مكان نحتاج لنكمل العدد إلى 10
        needed_count = target_count - len(recommendations)
        
        # اختيار أماكن عشوائية تكمل العدد
        extra_random_picks = random.sample(remaining_places, k=min(needed_count, len(remaining_places)))
        
        # دمج التوصيات الحقيقية مع العشوائية (التوصيات الحقيقية تظهر أولاً)
        full_recommendations = recommendations + extra_random_picks
        return full_recommendations

    # الحالة الثالثة: لو الكولابريتيف جاب 10 أو أكتر، نرجعهم زي ما هما (أو أول 10)
    return recommendations[:target_count]

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
