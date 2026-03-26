import random 
from fastapi import FastAPI
from database import get_live_data
from recommender import get_user_based_faiss

app = FastAPI(title="Lumeria AI API")

@app.get("/recommend/user/{user_id}")
def user_api(user_id: str):
    # 1. سحب البيانات
    ratings_df, places_df = get_live_data()
    
    # 2. محاولة جلب توصيات FAISS
    recommendations = get_user_based_faiss(user_id, ratings_df, places_df)
    
    # 3. لو مفيش توصيات (يوزر جديد) -> 10 أماكن عشوائية تماماً
    if not recommendations:
        # تحويل الداتا لـ list من القواميس
        all_places = places_df.to_dict(orient="records")
        
        # اختيار 10 أماكن عشوائيةأو أقل
        num_to_select = min(10, len(all_places))
        random_places = random.sample(all_places, k=num_to_select)
        
        return random_places
    
    return recommendations
