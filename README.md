# Curated Product Recommendation System

A Streamlit demo that recommends products from a starting product. It combines content-based TF-IDF similarity with collaborative signals from user ratings.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. The app includes demo catalog and rating files in `data/` and exports the visible shortlist to CSV.

## Test

```powershell
pytest -q
```

## Replace the data

`data/products.csv` requires `product_id`, `name`, `category`, `price`, `description`, and `image_url`. `data/ratings.csv` requires `user_id`, `product_id`, and `rating`.
