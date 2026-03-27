import random
from fastapi import FastAPI
from database import get_live_data
from recommender import get_user_based_faiss, get_popular_list

app = FastAPI(title="Lumeria AI API")

@app.get("/")
def root():
    return {"status": "Lumeria Recommendation Engine is Online"}

@app.get("/recommend/popular")
def popular_api(location: str = None):
    ratings_df, places_df = get_live_data()
    return get_popular_list(ratings_df, places_df, location)

@app.get("/recommend/user/{user_id}")
def user_api(user_id: str):
    # 1. سحب البيانات من سوبا بيز
    ratings_df, places_df = get_live_data()
    
    # 2. محاولة جلب توصيات بناءً على ذوق اليوزر (FAISS)
    recommendations = get_user_based_faiss(user_id, ratings_df, places_df)
    
    # 3. لو اليوزر جديد (النتائج فاضية) ->  أماكن عشوائية
    if not recommendations:
        # تحويل جدول الأماكن لقائمة من القواميس
        all_places_list = places_df.to_dict(orient="records")
        num_to_select = min(10, len(all_places_list))
        
        # اختيار الأماكن العشوائية
        random_recommendations = random.sample(all_places_list, k=num_to_select)
        
        return random_recommendations

    return recommendations
