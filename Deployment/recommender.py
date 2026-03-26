import numpy as np
import pandas as pd
import faiss

def get_popular_list(ratings_df, places_df, location=None):
    # دمج التقييمات مع تفاصيل الأماكن
    df = pd.merge(ratings_df, places_df, left_on="place_id", right_on="ID")
    
    # فلترة حسب المحافظة لو اليوزر طلب، غير كدة مصر كلها
    if location and location.strip() != "":
        df = df[df['Location'].str.contains(location, case=False, na=False)]
    
    if df.empty: return []

    # حساب الـ Weighted Rating (Bayesian Average)
    stats = df.groupby("Name")["rating"].agg(mean="mean", count="count").reset_index()
    C = stats['mean'].mean()
    m = stats['count'].quantile(0.8) if len(stats) > 5 else 1
    
    stats['score'] = (stats['count']/(stats['count']+m) * stats['mean']) + (m/(stats['count']+m) * C)
    
    # دمج النتائج مع البيانات الكاملة (صور، وصف، الخ)
    result = pd.merge(stats[['Name', 'score']], places_df, on="Name")
    return result.sort_values(by="score", ascending=False).head(10).to_dict(orient="records")

def get_user_based_faiss(user_id, ratings_df, places_df):
    # تحويل التقييمات لمصفوفة (Pivot Table)
    matrix = ratings_df.pivot(index='user_id', columns='place_id', values='rating').fillna(0)
    
    if user_id not in matrix.index: return []

    # استخدام FAISS للبحث عن "التوائم" (المستخدمين الشبيهين)
    data = matrix.values.astype('float32')
    faiss.normalize_L2(data) # لجعل الحسابات بدقة Cosine Similarity
    
    index = faiss.IndexFlatIP(data.shape[1]) 
    index.add(data)
    
    user_idx = matrix.index.get_loc(user_id)
    query_vec = data[user_idx:user_idx+1]
    
    # البحث عن أقرب 5 مستخدمين
    distances, indices = index.search(query_vec, k=min(5, len(matrix)))
    
    similar_users_ids = matrix.index[indices[0][1:]]
    
    # ترشيح أماكن قيمها هؤلاء المستخدمين بـ 4 أو 5 نجوم ولم يزرها المستخدم الحالي
    recommendations = ratings_df[ratings_df['user_id'].isin(similar_users_ids) & (ratings_df['rating'] >= 4)]
    user_visited = ratings_df[ratings_df['user_id'] == user_id]['place_id'].tolist()
    
    final_rec_ids = recommendations[~recommendations['place_id'].isin(user_visited)]['place_id'].unique()
    
    # إرجاع تفاصيل الأماكن المرشحة
    return places_df[places_df['ID'].isin(final_rec_ids)].head(5).to_dict(orient="records")