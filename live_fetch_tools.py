import os
import sys
import json
import asyncio
import logging
import re
from typing import List, Dict, Any
from pymongo import MongoClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

BACKEND_DIR = os.path.join(BASE_DIR, "Backend")
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from Backend.Services.product_hunt.website_crawler import ContentFetcher, LinkExtractor
from Backend.Services.product_hunt.content_cleaner import clean_html_to_markdown

logger = logging.getLogger(__name__)

def get_mongo_db():
    uri = os.getenv("MONGO_CONNECTION_STRING")
    if not uri:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        return client["influenze_ai_marketing"]
    except Exception:
        return None

def fetch_details_from_mongodb(product_name: str) -> Dict[str, Any]:
    """Check MongoDB for existing rich intelligence context."""
    db = get_mongo_db()
    if db is None:
        return {}
    
    # Case-insensitive regex search
    prod_query = {"name": re.compile(f"^{re.escape(product_name)}$", re.IGNORECASE)}
    product = db.product_intelligence.find_one(prod_query)
    
    if not product:
        # Fallback partial match
        prod_query_partial = {"name": re.compile(re.escape(product_name), re.IGNORECASE)}
        product = db.product_intelligence.find_one(prod_query_partial)
        
    if not product:
        return {}
        
    company_id = product.get("company_id")
    if not company_id:
        return {}
        
    company = db.companies.find_one({"company_id": company_id})
    if not company:
        return {}
        
    intel = company.get("intelligence", {})
    if not intel:
        return {}
        
    return {
        "product_name": product.get("name"),
        "company_name": company.get("name"),
        "website": company.get("website", product.get("website", "")),
        "company_overview": intel.get("company_overview", ""),
        "key_features": intel.get("key_features", []),
        "target_audience": intel.get("target_audience", []),
        "industry": company.get("industry", ""),
        "source": "MongoDB_Intelligence_Cache"
    }

async def crawl_product_website_async(url: str) -> Dict[str, str]:
    """
    Crawls a product's website, including its homepage and top subpages (pricing, features).
    Returns a dictionary of cleaned markdown contents.
    """
    fetcher = ContentFetcher(timeout=10)
    extractor = LinkExtractor()
    
    # 1. Fetch homepage
    homepage_html = fetcher.fetch_page_html(url)
    if not homepage_html:
        return {"error": "Failed to fetch homepage. Crawl blocked or timeout."}
    
    # 2. Extract internal links
    internal_links = extractor.extract_internal_links(homepage_html, url)
    
    # 3. Classify and pick top links
    candidates = extractor.filter_with_rule_engine(internal_links)
    
    priority = re.compile(r'(pricing|features|solutions|docs|about)', re.IGNORECASE)
    candidates.sort(key=lambda x: 0 if priority.search(x) else 1)
    
    selected_links = candidates[:3]
    
    raw_content = {"homepage": homepage_html}
    for sub_url in selected_links:
        try:
            html = fetcher.fetch_page_html(sub_url)
            if html:
                key = sub_url.split("/")[-1] or sub_url.split("/")[-2]
                raw_content[key] = html
        except Exception:
            pass
            
    # 4. Clean all to markdown
    clean_content = {}
    for page_name, html_source in raw_content.items():
        try:
            markdown = clean_html_to_markdown(html_source)
            clean_content[page_name] = markdown[:3000] # Truncate to save tokens
        except Exception:
            pass
            
    clean_content["source"] = "Live_Web_Crawl"
    return clean_content

async def crawl_multiple_websites(name_to_url: Dict[str, str]) -> Dict[str, Any]:
    """
    Async function to fetch deep context for multiple products.
    Checks MongoDB first. Falls back to live crawl if MongoDB doesn't have it.
    Awaited directly inside the agent's running event loop — no nested loops.
    """
    final_result = {}
    urls_to_crawl = []
    crawl_url_to_name = {}

    # 1. Check MongoDB first for all products
    for name, url in name_to_url.items():
        mongo_data = fetch_details_from_mongodb(name)
        if mongo_data and mongo_data.get("company_overview"):
            final_result[name] = mongo_data
        elif url and url.startswith("http"):
            urls_to_crawl.append(url)
            crawl_url_to_name[url] = name
        else:
            # No valid URL and no MongoDB data — return empty
            final_result[name] = {"error": f"No URL or MongoDB data available for '{name}'"}

    # 2. Await async crawls for anything not in MongoDB
    if urls_to_crawl:
        tasks = [crawl_product_website_async(url) for url in urls_to_crawl]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, res in zip(urls_to_crawl, results):
            name = crawl_url_to_name[url]
            if isinstance(res, Exception):
                final_result[name] = {"error": str(res)}
            else:
                final_result[name] = res

    return final_result
