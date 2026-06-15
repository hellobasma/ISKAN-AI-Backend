import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import json

def get_similar_properties(target_property_id, properties_df, top_n=4):
    """
    Find similar properties using a content-based recommendation approach.
    
    Args:
        target_property_id (str): The ID of the currently viewed property.
        properties_df (pd.DataFrame): DataFrame containing all properties.
        top_n (int): Number of similar properties to return.
        
    Returns:
        str: JSON string containing the recommended properties.
    """
    # 1. Validate Target Property
    if target_property_id not in properties_df['property_id'].values:
        return json.dumps({"similar_properties": []})
        
    target_prop = properties_df[properties_df['property_id'] == target_property_id].iloc[0]
    target_price = target_prop['price_val']
    
    # 2. Filtering Constraints
    # Exclude the target property itself
    candidates_df = properties_df[properties_df['property_id'] != target_property_id].copy()
    
    # Filter by price (+/- 20% limit)
    min_price = target_price * 0.80
    max_price = target_price * 1.20
    candidates_df = candidates_df[(candidates_df['price_val'] >= min_price) & (candidates_df['price_val'] <= max_price)]
    
    if candidates_df.empty:
        return json.dumps({"similar_properties": []})
        
    # Re-combine target and candidates for consistent preprocessing (handling unseen categories safely)
    # In a production system, you might pre-compute features or use a saved pipeline
    analysis_df = pd.concat([pd.DataFrame([target_prop]), candidates_df]).reset_index(drop=True)
    
    # 3. Feature Engineering
    # Define features for the model
    categorical_features = ['location', 'property_type']
    numerical_features = ['price_val', 'bedrooms', 'area_val']
    
    # Scale numerical features (Min-Max Scaling)
    scaler = MinMaxScaler()
    num_scaled = scaler.fit_transform(analysis_df[numerical_features])
    num_scaled_df = pd.DataFrame(num_scaled, columns=numerical_features)
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded = encoder.fit_transform(analysis_df[categorical_features])
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(categorical_features))
    
    # Combine engineered features
    features_df = pd.concat([num_scaled_df, cat_encoded_df], axis=1)
    
    # 4. Apply Feature Weights
    # "Ensure that 'Location', 'Property Type', and 'Price' have a higher impact"
    weights = {
        'price_val': 3.0,
        'bedrooms': 1.0,
        'area_val': 1.0,
    }
    
    for col in features_df.columns:
        if col.startswith('location_'):
            features_df[col] *= 3.0  # Higher weight for location
        elif col.startswith('property_type_'):
            features_df[col] *= 3.0  # Higher weight for property type
        elif col in weights:
            features_df[col] *= weights[col]
            
    # 5. Similarity Metric Calculation (Cosine Similarity)
    # Target property is at index 0 after re-combining
    target_vector = features_df.iloc[0:1]
    candidate_vectors = features_df.iloc[1:]
    
    # Compute similarity distances
    similarities = cosine_similarity(target_vector, candidate_vectors)[0]
    
    # 6. Sort and Select Top N
    candidates_df['similarity'] = similarities
    top_matches = candidates_df.sort_values(by='similarity', ascending=False).head(top_n)
    
    # 7. Format Output as JSON (exclude similarity scores/percentages)
    results = []
    for _, row in top_matches.iterrows():
        results.append({
            "property_id": str(row['property_id']),
            "title": row['title'],
            "price": row['price_display'],
            "bedrooms": int(row['bedrooms']),
            "area": row['area_display'],
            "thumbnail_url": row['thumbnail_url']
        })
        
    return {"recommendations": results}

if __name__ == "__main__":
    # 1. Data Architecture (Mock Data Definition)
    # This dummy dataset defines the expected structure:
    # We maintain numeric columns ('_val') for computation and string columns ('_display') for the UI output
    mock_data = [
        {
            "property_id": "1045", "title": "Apartment in Tanta", "location": "Tanta", 
            "property_type": "Apartment", "price_val": 1200000, "price_display": "1,200,000 EGP",
            "bedrooms": 3, "area_val": 120, "area_display": "120 sqm", "thumbnail_url": "/images/prop_1045.jpg"
        },
        {
            "property_id": "890", "title": "Apartment in Tanta", "location": "Tanta", 
            "property_type": "Apartment", "price_val": 1150000, "price_display": "1,150,000 EGP",
            "bedrooms": 3, "area_val": 115, "area_display": "115 sqm", "thumbnail_url": "/images/prop_890.jpg"
        },
        {
            "property_id": "1003", "title": "Luxury Villa in Cairo", "location": "Cairo", 
            "property_type": "Villa", "price_val": 8000000, "price_display": "8,000,000 EGP",
            "bedrooms": 5, "area_val": 400, "area_display": "400 sqm", "thumbnail_url": "/images/prop_1003.jpg"
        },
        {
            "property_id": "1004", "title": "Studio in Alexandria", "location": "Alexandria", 
            "property_type": "Studio", "price_val": 900000, "price_display": "900,000 EGP",
            "bedrooms": 1, "area_val": 60, "area_display": "60 sqm", "thumbnail_url": "/images/prop_1004.jpg"
        },
        {
            "property_id": "1005", "title": "Spacious Apartment in Tanta", "location": "Tanta", 
            "property_type": "Apartment", "price_val": 1300000, "price_display": "1,300,000 EGP",
            "bedrooms": 4, "area_val": 150, "area_display": "150 sqm", "thumbnail_url": "/images/prop_1005.jpg"
        },
        {
            "property_id": "1006", "title": "Apartment in Mansoura", "location": "Mansoura", 
            "property_type": "Apartment", "price_val": 1250000, "price_display": "1,250,000 EGP",
            "bedrooms": 3, "area_val": 125, "area_display": "125 sqm", "thumbnail_url": "/images/prop_1006.jpg"
        }
    ]
    
    df = pd.DataFrame(mock_data)
    
    print("--- Testing ISKAN Recommender System ---")
    target_id = "1045"
    print(f"Target Property (Currently Viewing): {target_id}")
    print(df[df['property_id'] == target_id][['title', 'location', 'price_display']].to_string(index=False))
    
    print("\nFetching Similar Properties...\n")
    
    # Run the main recommendation function
    result_json = get_similar_properties(target_property_id=target_id, properties_df=df, top_n=4)
    
    print("--- Output JSON ---")
    print(result_json)
