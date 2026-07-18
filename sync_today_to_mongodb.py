import os
import sys
import json
import re
import datetime
import sqlite3
from pymongo import MongoClient
from dotenv import load_dotenv
from openai import AzureOpenAI

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

LOCAL_ENV = os.path.join(SCRIPT_DIR, ".env")
if os.path.exists(LOCAL_ENV):
    ENV_PATH = LOCAL_ENV
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ENV_PATH = os.path.join(BASE_DIR, "Backend", ".env")
load_dotenv(dotenv_path=ENV_PATH)

CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")
DB_PATH = os.path.join(SCRIPT_DIR, "knowledge_graph.db")

def get_mongo_db():
    uri = os.getenv("MONGO_CONNECTION_STRING")
    if not uri:
        raise ValueError("MONGO_CONNECTION_STRING is not set in environment.")
    client = MongoClient(uri)
    return client["influenze_ai_marketing"]

def clean_text_id(text):
    if not text:
        return ""
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower().strip()

def enrich_product_with_ai(client, deployment_name, name, tagline, description, categories):
    """Call Azure OpenAI to get overview, features, and target audience based on PH metadata."""
    prompt = f"""
Analyze this Product Hunt launch and return a JSON object with:
1. "company_overview": a 1-2 sentence overview of what they do.
2. "key_features": a list of 2-4 key features.
3. "target_audience": a list of 2-3 target user segments (e.g. "Developers", "Marketing Teams", "Content Creators", "Shopify Store Owners", "Founders").

Product Info:
Name: {name}
Tagline: {tagline}
Description: {description}
Categories: {', '.join(categories)}

Output strictly in JSON format matching this schema:
{{
  "company_overview": "string",
  "key_features": ["string"],
  "target_audience": ["string"]
}}
"""
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that extracts product intelligence. Output raw JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        # Clean up code blocks if present
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        return json.loads(content)
    except Exception as e:
        print(f"  [AI Error] Failed to enrich {name}: {e}")
        return {
            "company_overview": tagline,
            "key_features": [tagline],
            "target_audience": ["General Users"]
        }

def main():
    print("=== SYNC TODAY'S LAUNCHES TO MONGODB & REBUILD KG ===")
    
    # 1. Connect to MongoDB
    try:
        db = get_mongo_db()
        print("Connected to MongoDB successfully.")
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        return

    # 2. Setup OpenAI
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if not all([api_key, endpoint, deployment_name]):
        print("Error: Azure OpenAI variables are not fully set in Backend/.env")
        return
        
    openai_client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)

    # 3. Read today's cache file
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    cache_file = os.path.join(CACHE_DIR, f"products_{today_str}.json")
    if not os.path.exists(cache_file):
        print(f"Today's cache file '{cache_file}' not found. Run the Streamlit app first to generate it.")
        return

    with open(cache_file, "r", encoding="utf-8") as f:
        products = json.load(f)

    print(f"Loaded {len(products)} products from today's cache.")

    for i, p in enumerate(products, 1):
        name = p.get("name")
        product_id = p.get("id")
        tagline = p.get("tagline", "")
        description = p.get("description", "")
        website = p.get("website", "")
        
        # Get categories
        categories = p.get("topics", [])

        company_id = clean_text_id(name)
        if not company_id:
            company_id = f"co_ph_{product_id}"

        print(f"[{i}/{len(products)}] Processing '{name}' (Company ID: {company_id})...")

        # Check if already enriched in MongoDB companies collection
        existing_company = db.companies.find_one({"company_id": company_id})
        if existing_company and existing_company.get("intelligence", {}).get("company_overview"):
            print(f"  → Found existing intelligence in MongoDB. Skipping enrichment.")
            intelligence = existing_company.get("intelligence")
        else:
            print(f"  → Calling AI to enrich product details...")
            intelligence = enrich_product_with_ai(openai_client, deployment_name, name, tagline, description, categories)

        # Upsert Company
        db.companies.update_one(
            {"company_id": company_id},
            {"$set": {
                "company_id": company_id,
                "name": name,
                "website": website,
                "industry": "",
                "intelligence": intelligence
            }},
            upsert=True
        )

        # Upsert Product Intelligence
        db.product_intelligence.update_one(
            {"product_hunt_id": product_id},
            {"$set": {
                "product_hunt_id": product_id,
                "name": name,
                "company_id": company_id,
                "website": website,
                "tagline": tagline,
                "categories": categories,
                "votes": p.get("votes", 0),
                "product_hunt_url": f"https://www.producthunt.com/posts/{product_id}"
            }},
            upsert=True
        )
        print("  → Upserted to MongoDB successfully.")

    print("\nAll products synced to MongoDB.")
    
    # 4. Run build_kg.py to rebuild the local SQLite knowledge_graph.db
    print("\nRebuilding the Knowledge Graph database...")
    try:
        import build_kg
        build_kg.main()
        print("Knowledge Graph database rebuilt successfully!")
    except Exception as e:
        print(f"Error rebuilding Knowledge Graph: {e}")

if __name__ == "__main__":
    main()
