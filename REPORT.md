# Product Recommendation System Report

## Objective

Recommend relevant products from a selected product and optionally personalize the results for a known user.

## Data

The demo uses a small catalog in `data/products.csv` and explicit 1-5 user ratings in `data/ratings.csv`. Product text combines name, category, and description. Image URLs provide thumbnails in the UI.

## Method

1. Validate required catalog and rating columns.
2. Vectorize product text with scikit-learn TF-IDF and calculate cosine similarity.
3. Build a user-product rating matrix with pandas.
4. Generate collaborative scores from highly rated items for a selected user, or from users who rated the selected product when no user is chosen.
5. Blend content similarity (65%) and collaborative score (35%), remove the input product, and return the top N items.
6. Render product image, category, price, description, match score, and recommendation reason in Streamlit.

The content component is the fallback for sparse users and cold-start interactions. In a production system, the blend weights should be tuned against offline metrics such as precision@k, recall@k, and NDCG@k.

## Limitations and next steps

The fixture is intentionally small and the image links are external demo assets. A production deployment should add catalog freshness, missing-image fallbacks, implicit events such as clicks and purchases, popularity priors, filtering for inventory, and an evaluation split based on time.

## Deliverables

- `app.py`: Streamlit user interface and CSV export.
- `recommender.py`: reusable hybrid recommendation engine.
- `data/products.csv`, `data/ratings.csv`: test/demo data.
- `test_recommender.py`: focused behavior tests.
