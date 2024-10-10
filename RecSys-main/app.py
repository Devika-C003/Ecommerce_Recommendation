from flask import Flask, request, render_template, session, redirect, url_for, jsonify
from flask_session import Session
import pandas as pd
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Configure session to use filesystem (server-side session)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load files
trending_products = pd.read_csv("models/trending_new.csv")
all_products_data = pd.read_csv("models/data_s2.csv")

# Truncate function
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

# Helper functions and recommendations

def get_products(data, index, n):
    return data[index:index+n]

# Rating-Based Recommendation
def rating_based_recommendation(data, top_n=10):
    # Calculate average ratings, sort, and select top n items
    top_rated_items = (
        data.groupby(['Name', 'ReviewCount', 'Brand', 'ImageURL'])['Rating']
        .mean().reset_index()
        .sort_values(by=['Rating', 'ReviewCount'], ascending=[False, False])
        .head(top_n)
    )

    # Convert to integer and merge to get all columns, then select necessary columns
    top_rated_items[['Rating', 'ReviewCount']] = top_rated_items[['Rating', 'ReviewCount']]
    return top_rated_items.merge(data, on=['Name', 'Rating', 'ReviewCount', 'Brand', 'ImageURL'], how='left')[
        ['Name', 'ImageURL', 'Brand', 'Rating', 'ReviewCount', 'Description', 'Price']
    ]
    
# Content-Based Recommendations
def content_based_recommendations(data, item_name, top_n=10):
    # Check if the item name exists in the training data
    if item_name not in data['Name'].values:
        print(f"Item '{item_name}' not found in the training data.")
        return pd.DataFrame()

    # Create a TF-IDF vectorizer for item descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Apply TF-IDF vectorization to item descriptions
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(data['Tags'])

    # Calculate cosine similarity between items based on descriptions
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    # Find the index of the item
    item_index = data[data['Name'] == item_name].index[0]

    # Get the cosine similarity scores for the item
    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    # Sort similar items by similarity score in descending order
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    # Get the top N most similar items (excluding the item itself)
    top_similar_items = similar_items[1:top_n+1]

    # Get the indices of the top similar items
    recommended_item_indices = [x[0] for x in top_similar_items]

    # Get the details of the top similar items
    recommended_items_details = data.iloc[recommended_item_indices][['Name', 'ImageURL', 'Brand', 'Rating', 'ReviewCount', 'Description', 'Price']]
    # recommended_items_details = data.iloc[recommended_item_indices]

    return recommended_items_details

# Collaborative Filtering Recommendations
def collaborative_filtering_recommendations(data, target_user_id, top_n=10):
    # Create the user-item matrix
    user_item_matrix = data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)

    # Calculate the user similarity matrix using cosine similarity
    user_similarity = cosine_similarity(user_item_matrix)

    # Find the index of the target user in the matrix
    target_user_index = user_item_matrix.index.get_loc(target_user_id)

    # Get the similarity scores for the target user
    user_similarities = user_similarity[target_user_index]

    # Sort the users by similarity in descending order (excluding the target user)
    similar_users_indices = user_similarities.argsort()[::-1][1:]

    # Generate recommendations based on similar users
    recommended_items = []

    for user_index in similar_users_indices:
        # Get items rated by the similar user but not by the target user
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (rated_by_similar_user == 0) & (user_item_matrix.iloc[target_user_index] == 0)

        # Extract the item IDs of recommended items
        recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])

    # Get the details of recommended items
    recommended_items_details = data[data['ProdID'].isin(recommended_items)][['Name', 'ImageURL', 'Brand', 'Rating', 'ReviewCount', 'Description', 'Price']]

    return recommended_items_details.head(top_n)

# Hybrid Recommendations (Combine Content-Based and Collaborative Filtering)
def hybrid_recommendations(data, target_user_id, item_name, top_n=10):
    # Get content-based recommendations
    content_based_rec = content_based_recommendations(data, item_name, top_n)

    # Get collaborative filtering recommendations
    collaborative_filtering_rec = collaborative_filtering_recommendations(data, target_user_id, top_n)
    
    # Merge and deduplicate the recommendations
    hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec]).drop_duplicates()
    
    return hybrid_rec.head(top_n)


@app.route("/")
def index():
    user = 4
    cf_recommendations = collaborative_filtering_recommendations(all_products_data, user, 8)
    trending_products = rating_based_recommendation(all_products_data)
    return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate, cf_recommendations=cf_recommendations)

@app.route("/all-products")
def all_products():
    search_query = request.args.get('search')
    if search_query:
        filtered_products = all_products_data[all_products_data.apply(lambda row: search_query.lower() in row.astype(str).str.lower().values, axis=1)]
    else:
        filtered_products = get_products(all_products_data, 0, 20)
        
    return render_template('all_products.html', all_products=filtered_products, truncate=truncate)

@app.route("/product/<int:product_id>")
def product_detail(product_id):
    product = all_products_data.loc[product_id]
    similar_products = content_based_recommendations(all_products_data, product['Name'], 8)
    return render_template('product.html', product=product, similar_products=similar_products, truncate=truncate)

@app.route("/add-to-cart", methods=["POST"])
def add_to_cart():
    product_id = request.json.get("product_id")
    product = all_products_data.loc[product_id]
    cart = session.get("cart", [])
    cart.append(product_id)
    session["cart"] = cart
    return jsonify({"message": "Added to cart!"})

@app.route("/cart")
def cart():
    cart = session.get("cart", [])
    cart_products = all_products_data.loc[cart]
    total_price = cart_products["Price"].sum()

    recommendations = []
    for product_id in cart:
        product = all_products_data.loc[product_id]
        product_recommendations = hybrid_recommendations(all_products_data, 4, product["Name"], 2)
        recommendations.append(product_recommendations)

    return render_template('cart.html', cart_products=cart_products, total_price=total_price, recommendations=recommendations, truncate=truncate)

@app.route("/remove-from-cart/<int:product_id>", methods=["POST"])
def remove_from_cart(product_id):
    cart = session.get("cart", [])
    cart.remove(product_id)
    session["cart"] = cart
    return redirect(url_for("cart"))

if __name__ == '__main__':
    app.run(debug=True)
