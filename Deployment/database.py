import os
import pandas as pd
from supabase import create_client

URL = os.environ.get("SUPABASE_URL")
KEY = os.environ.get("SUPABASE_KEY")

supabase = create_client(URL, KEY)

def get_live_data() -> None:  
    res_ratings = supabase.table("user_ratings").select("user_id, place_id, rating").execute()
    res_places = supabase.table("places").select("*").execute()
    
    ratings_df = pd.DataFrame(res_ratings.data)
    places_df = pd.DataFrame(res_places.data)
    
    return ratings_df, places_df.sort_values(by="ID")