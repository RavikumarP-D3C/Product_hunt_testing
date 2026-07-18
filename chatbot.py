import os
import sys
import json
import sqlite3
import datetime
import tempfile
import asyncio
import streamlit as st
from dotenv import load_dotenv

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

BACKEND_DIR = os.path.join(BASE_DIR, "Backend")
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from Backend.Utils.llm_client import get_llm_client
from Backend.Services.product_hunt.fetch_products import fetch_latest_products
from live_fetch_tools import crawl_multiple_websites
from kg_tools import (
    query_kg_for_product,
    find_market_gaps,
    find_uncontested_products,
    cross_reference_today_vs_kg,
)

ENV_PATH = os.path.join(BASE_DIR, ".env")
if not os.path.exists(ENV_PATH):
    ENV_PATH = os.path.join(BASE_DIR, "Backend", ".env")
load_dotenv(dotenv_path=ENV_PATH)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_graph.db")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

st.set_page_config(page_title="Product Hunt AI Assistant", layout="wide", initial_sidebar_state="expanded")


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────

from pymongo import MongoClient
import re

def get_mongo_db():
    uri = os.getenv("MONGO_CONNECTION_STRING")
    if not uri:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        return client["influenze_ai_marketing"]
    except Exception:
        return None

def clean_text_id(text):
    if not text:
        return ""
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower().strip()

from openai import AzureOpenAI

def enrich_product_with_ai(name, tagline, description, categories):
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if not all([api_key, endpoint, deployment_name]):
        return {
            "company_overview": tagline,
            "key_features": [tagline],
            "target_audience": ["General Users"]
        }
        
    client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
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
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        return json.loads(content)
    except Exception as e:
        print(f"[AI Error] Failed to enrich {name}: {e}")
        return {
            "company_overview": tagline,
            "key_features": [tagline],
            "target_audience": ["General Users"]
        }

def matches_domains(product: dict) -> bool:
    keywords = {
        "social media": ["social", "media", "marketing", "instagram", "twitter", "linkedin", "tiktok", "content creation", "creator"],
        "developer tools": ["developer", "devtools", "coding", "github", "sandbox", "database", "api", "terminal", "ide", "vscode", "deploy"],
        "ai agent": ["agent", "ai agent", "autonomous", "copilot", "assistant", "llm", "rag", "ai-powered"]
    }
    name = product.get("name", "").lower()
    tagline = product.get("tagline", "").lower()
    description = product.get("description", "").lower()
    topics = [t.lower() for t in product.get("topics", [])]
    
    text = f"{name} {tagline} {description} {' '.join(topics)}"
    for domain, domain_keywords in keywords.items():
        for kw in domain_keywords:
            if kw in text:
                return True
    return False

def sync_and_rebuild_today_launches():
    print("[Sync] Fetching today's raw products from Product Hunt...")
    try:
        raw_products = fetch_latest_products(limit=40)
    except Exception as e:
        print(f"[Sync Error] Error fetching from Product Hunt: {e}")
        return
        
    print("[Sync] Filtering products to keep: Social Media, Developer Tools, AI Agents...")
    filtered_products = []
    for p in raw_products:
        if matches_domains(p):
            filtered_products.append(p)
            
    print(f"[Sync] Found {len(filtered_products)} products matching the specified domains (out of {len(raw_products)} total launches).")
    if not filtered_products:
        print("[Sync] No launches match the specified domains today.")
        return
        
    print("[Sync] Connecting to MongoDB Cloud Database...")
    db = get_mongo_db()
    if db is None:
        print("[Sync Error] Error: Could not connect to MongoDB.")
        return
        
    for i, p in enumerate(filtered_products, 1):
        name = p.get("name")
        product_id = p.get("id")
        tagline = p.get("tagline", "")
        description = p.get("description", "")
        website = p.get("website", "")
        categories = p.get("topics", [])
        
        company_id = clean_text_id(name)
        if not company_id:
            company_id = f"co_ph_{product_id}"
            
        print(f"[Sync] [{i}/{len(filtered_products)}] Processing '{name}'...")
        
        # Check MongoDB
        existing_company = db.companies.find_one({"company_id": company_id})
        if existing_company and existing_company.get("intelligence", {}).get("company_overview"):
            intelligence = existing_company.get("intelligence")
        else:
            intelligence = enrich_product_with_ai(name, tagline, description, categories)
            
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
        
    print("[Sync] Rebuilding local SQLite Knowledge Graph (knowledge_graph.db)...")
    try:
        from build_kg import main as rebuild_kg_main
        rebuild_kg_main()
        print("[Sync] Knowledge Graph rebuilt successfully!")
    except Exception as e:
        print(f"[Sync Error] Error rebuilding Knowledge Graph: {e}")
        return
        
    # Save the filtered products to today's products cache
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    cache_file = os.path.join(CACHE_DIR, f"products_{today_str}.json")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(filtered_products, f, indent=4)
        
    print("[Sync] Today's filtered launches successfully synchronized!")

def run_async_in_thread(coro):
    import threading
    res_list = []
    err_list = []
    def run():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(coro)
            res_list.append(res)
        except Exception as e:
            err_list.append(e)
        finally:
            loop.close()
    t = threading.Thread(target=run)
    t.start()
    t.join()
    if err_list:
        raise err_list[0]
    return res_list[0]


def get_today_products():
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    cache_file = os.path.join(CACHE_DIR, f"products_{today_str}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    with st.spinner(f"Fetching fresh products from Product Hunt for {today_str}..."):
        try:
            products = fetch_latest_products(limit=30)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(products, f, indent=4)
            return products
        except Exception as e:
            st.error(f"Error fetching from Product Hunt API: {e}")
            return []


def get_kg_summary_context():
    """Quick top-level summary of KG — passed in every system prompt."""
    if not os.path.exists(DB_PATH):
        return "Knowledge Graph database not found."
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT entity_type, COUNT(*) as cnt FROM knowledge_entities GROUP BY entity_type")
        entity_counts = {row[0]: row[1] for row in cur.fetchall()}
        cur.execute("SELECT relationship_type, COUNT(*) as cnt FROM knowledge_relationships GROUP BY relationship_type")
        rel_counts = {row[0]: row[1] for row in cur.fetchall()}
        cur.execute("SELECT canonical_name FROM knowledge_entities WHERE entity_type='Category' LIMIT 30")
        categories = [row[0] for row in cur.fetchall()]
        cur.execute("SELECT canonical_name FROM knowledge_entities WHERE entity_type='Product' LIMIT 10")
        sample_products = [row[0] for row in cur.fetchall()]
        conn.close()
        ctx = "### Knowledge Graph Summary:\n"
        ctx += f"Entities: {entity_counts}\n"
        ctx += f"Relationships: {rel_counts}\n"
        ctx += f"Known Categories: {', '.join(categories)}\n"
        ctx += f"Sample Products: {', '.join(sample_products)}\n"
        return ctx
    except Exception as e:
        return f"KG error: {e}"


def lookup_urls(product_names: list, today_products: list) -> dict:
    results = {}
    name_lower_to_url = {p.get("name", "").lower(): p.get("website", "") for p in today_products}
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT canonical_name, attributes_json FROM knowledge_entities WHERE entity_type='Product'")
            for row in cur.fetchall():
                attrs = json.loads(row[1] or "{}")
                name_lower_to_url[row[0].lower()] = attrs.get("website", "")
            conn.close()
        except:
            pass
    for name in product_names:
        url = name_lower_to_url.get(name.lower(), "mongo_lookup_only")
        results[name] = url
    return results


# ─────────────────────────────────────────────────────────────
# Tool Execution Functions
# ─────────────────────────────────────────────────────────────

async def tool_get_deep_product_details(product_names: list, today_products: list) -> str:
    url_mapping = lookup_urls(product_names, today_products)
    st.toast(f"Fetching deep intelligence for: {', '.join(url_mapping.keys())}")
    crawled_data = await crawl_multiple_websites(url_mapping)
    parts = []
    for name, data in crawled_data.items():
        parts.append(f"### Deep Details: {name}\n{json.dumps(data, indent=2)}")
    return "\n\n".join(parts)


def tool_query_kg_for_product(product_name: str) -> str:
    st.toast(f"Querying Knowledge Graph for: {product_name}")
    result = query_kg_for_product(product_name)
    return json.dumps(result, indent=2)


def tool_find_market_gaps(category: str) -> str:
    st.toast(f"Analyzing market gaps for category: {category}")
    result = find_market_gaps(category)
    return json.dumps(result, indent=2)


def tool_find_uncontested_products(today_products: list) -> str:
    st.toast("Checking today's products against Knowledge Graph for uncontested niches...")
    result = find_uncontested_products(today_products)
    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────────────────────
# OpenAI Tools Schema — 4 tools
# ─────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_deep_product_details",
            "description": (
                "Fetch live, comprehensive details (company overview, key features, "
                "pricing, target audience) for one or more products. Use when the user "
                "asks for 'full details', 'features', 'pricing', or 'company info'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of product names to fetch full details for.",
                    }
                },
                "required": ["product_names"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_kg_for_product",
            "description": (
                "Query the local Knowledge Graph for a specific product. Returns its "
                "features (HAS_FEATURE), target audience (TARGETS), categories (BELONGS_TO), "
                "competitors (COMPETES_WITH) with evidence + confidence scores, "
                "and similar products (SIMILAR_TO). Use when the user asks about "
                "KG relationships, competitors, evidence, or inference types for a product."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "Exact or partial name of the product to look up.",
                    }
                },
                "required": ["product_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_market_gaps",
            "description": (
                "Analyse the Knowledge Graph to detect market gaps in a product category. "
                "Returns all products in the category, their features, and the customer "
                "segments that are underserved (targeted by fewer than half the products). "
                "Use for questions about market gaps, underserved audiences, or 'what is "
                "missing' in a category."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Category to analyse (e.g. 'Developer Tools', 'SEO', 'Social Media').",
                    }
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_uncontested_products",
            "description": (
                "Check today's Product Hunt launches against the Knowledge Graph and "
                "identify which products have NO competitors or similar products in the graph. "
                "These are potentially novel niches or new market opportunities. "
                "Use when the user asks about 'new niches', 'uncontested markets', or "
                "'which products have no competitors'."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cross_reference_today_vs_kg",
            "description": (
                "Cross-reference today's Product Hunt products against the historical KG. "
                "Use this INSTEAD OF find_market_gaps when the user asks which of TODAY'S products "
                "match or fill gaps in a KG category. "
                "This tool ONLY returns today's products — it NEVER lists historical KG products as today's launches. "
                "Optionally provide target_category to also get that category's gap analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_category": {
                        "type": "string",
                        "description": "Optional KG category to analyse gaps for (e.g. 'SEO', 'Developer Tools').",
                    }
                },
                "required": [],
            },
        },
    },
]

def tool_cross_reference_today_vs_kg(today_products: list, target_category: str = None) -> str:
    st.toast("Cross-referencing today's products against KG (no data mixing)...")
    result = cross_reference_today_vs_kg(today_products, target_category or None)
    return json.dumps(result, indent=2)




async def run_responses_agent(
    system_prompt: str,
    user_query: str,
    tools: list,
    previous_response_id: str,
    today_products: list
) -> tuple[str, str]:
    """
    Executes a multi-turn, tool-calling loop using the Azure OpenAI Responses API.
    Returns (response_text, new_response_id).
    """
    client = get_llm_client()
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    # Flatten tools format for Azure Responses API
    flat_tools = []
    for t in tools:
        if "function" in t:
            flat_tools.append({
                "type": "function",
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "parameters": t["function"]["parameters"]
            })
        else:
            flat_tools.append(t)

    kwargs = {
        "model": deployment_name,
        "tools": flat_tools,
        "parallel_tool_calls": False,
        "stream": True
    }
    
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id
        kwargs["input"] = [{"role": "user", "content": user_query}]
    else:
        kwargs["input"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
    current_response_id = previous_response_id
    
    while True:
        response = await client.responses.create(**kwargs)
        
        accumulated_text = ""
        tool_call_to_execute = None
        
        async for event in response:
            if hasattr(event, 'response') and hasattr(event.response, 'id'):
                current_response_id = event.response.id
                
            if event.type == "response.output_text.delta":
                accumulated_text += event.delta
                
            elif event.type == "response.output_item.done":
                if hasattr(event, 'item') and hasattr(event.item, 'type') and event.item.type == 'function_call':
                    tool_call_to_execute = {
                        "name": event.item.name,
                        "arguments": json.loads(event.item.arguments),
                        "call_id": event.item.call_id
                    }
            elif event.type == "response.done":
                break
                
        if not tool_call_to_execute:
            return accumulated_text, current_response_id
            
        # Execute the tool
        fn = tool_call_to_execute["name"]
        args = tool_call_to_execute["arguments"]
        call_id = tool_call_to_execute["call_id"]
        
        if fn == "get_deep_product_details":
            tool_result = await tool_get_deep_product_details(args["product_names"], today_products)
        elif fn == "query_kg_for_product":
            tool_result = tool_query_kg_for_product(args["product_name"])
        elif fn == "find_market_gaps":
            tool_result = tool_find_market_gaps(args["category"])
        elif fn == "find_uncontested_products":
            tool_result = tool_find_uncontested_products(today_products)
        elif fn == "cross_reference_today_vs_kg":
            args_cat = args.get("target_category")
            tool_result = tool_cross_reference_today_vs_kg(today_products, args_cat)
        else:
            tool_result = f"Unknown tool: {fn}"
            
        # Submit the tool output
        kwargs = {
            "model": deployment_name,
            "previous_response_id": current_response_id,
            "input": [{
                "type": "function_call_output",
                "call_id": call_id,
                "output": tool_result
            }],
            "stream": True,
            "tools": flat_tools,
            "parallel_tool_calls": False
        }


async def generate_kg_response(user_query: str, today_products: list, kg_summary: str, previous_response_id: str = None) -> tuple[str, str]:
    today_context = "### Today's Live Products (from daily cache):\n"
    for p in today_products:
        today_context += f"- {p.get('name')}: {p.get('tagline')} | {p.get('website','')}\n"

    system_prompt = f"""You are an intelligent Product Hunt AI Assistant with 5 tools:

1. get_deep_product_details       - Fetch full company details, features, pricing from MongoDB or live web
2. query_kg_for_product           - Get ALL KG edges for a product (features, competitors, audience, evidence + confidence scores)
3. find_market_gaps               - Detect underserved customer segments and missing features IN THE HISTORICAL KG for a category
4. find_uncontested_products      - Identify today's products that have no KG competitors
5. cross_reference_today_vs_kg    - Properly link today's products to KG categories WITHOUT mixing historical KG data

=== CRITICAL DATA SOURCE RULES (NEVER VIOLATE) ===
- TODAY'S PRODUCTS come ONLY from the cache list below. They are today's real new launches.
- HISTORICAL KG PRODUCTS are from the SQLite Knowledge Graph (177 products). These are PAST launches.
- find_market_gaps() returns HISTORICAL KG products. NEVER claim those are "today's launches".
- cross_reference_today_vs_kg() is the ONLY tool that correctly links today's products to KG categories.
- If asked "which of today's products are in category X?", use cross_reference_today_vs_kg — NOT find_market_gaps.
- Always state the data source: "from today's cache" vs "from the historical Knowledge Graph".

=== TOOL USAGE RULES ===
- "full details / features / pricing / company info" → get_deep_product_details
- "why compete / competitors / evidence / confidence / inference_type" → query_kg_for_product  
- "market gap / underserved / what's missing in [category]" → find_market_gaps
- "novel niche / no competitors / uncontested" → find_uncontested_products
- "which of today's products are in [category] / cross-reference today vs KG" → cross_reference_today_vs_kg
- Always cite evidence_text and confidence scores when discussing KG relationships.

{today_context}

{kg_summary}
"""
    try:
        return await run_responses_agent(
            system_prompt=system_prompt,
            user_query=user_query,
            tools=TOOLS,
            previous_response_id=previous_response_id,
            today_products=today_products
        )
    except Exception as e:
        return f"Azure OpenAI API Error: {str(e)}", previous_response_id


def get_all_db_products() -> list:
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT canonical_name FROM knowledge_entities WHERE entity_type='Product'")
        products = [row[0] for row in cur.fetchall()]
        conn.close()
        return products
    except Exception:
        return []


async def generate_baseline_response(user_query: str, today_products: list, all_products: list, previous_response_id: str = None) -> tuple[str, str]:
    """Generates a response WITHOUT Knowledge Graph (baseline)."""
    today_context = "### Today's Live Products (from daily cache):\n"
    for p in today_products:
        today_context += f"- {p.get('name')}: {p.get('tagline')} | {p.get('website','')}\n"

    system_prompt = f"""You are an intelligent Product Hunt AI Assistant with 1 tool:

1. get_deep_product_details       - Fetch full company details, features, pricing from MongoDB or live web

=== ROLE & CAPABILITIES ===
- You are a Product Hunt AI Assistant helping users analyze daily Product Hunt launches and database products.
- You can answer questions based on your general knowledge, the daily cache of Product Hunt products, and the known database products listed below.
- If a user asks about any of the "Known Database Products" (e.g., "Mora Marketer"), you MUST call `get_deep_product_details` to fetch its features, target audience, and overview.
- You DO NOT have access to a historical database of relationships or Knowledge Graph (KG). You cannot query structured historical relationships, historical competitors, or deep market gaps.
- When asked about competitors or market gaps, do your best using only your general knowledge and whatever you can crawl/fetch. Always make it clear that you are guessing competitors based on general knowledge because you lack a Knowledge Graph.

=== DATA SOURCE RULES ===
- TODAY'S PRODUCTS come ONLY from the cache list below.
- KNOWN DATABASE PRODUCTS are historical launches. You do not know their details unless you call `get_deep_product_details`.

=== Known Database Products ===
{', '.join(all_products)}

=== Today's Live Products (from daily cache) ===
{today_context}
"""
    try:
        baseline_tools = [TOOLS[0]]
        return await run_responses_agent(
            system_prompt=system_prompt,
            user_query=user_query,
            tools=baseline_tools,
            previous_response_id=previous_response_id,
            today_products=today_products
        )
    except Exception as e:
        return f"Azure OpenAI API Error: {str(e)}", previous_response_id


# ─────────────────────────────────────────────────────────────
# Sidebar — Knowledge Graph Explorer
# ─────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.title("Knowledge Graph Explorer")
    st.sidebar.markdown("View Knowledge Graph stats and sync options.")

    # Quick KG stats
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT entity_type, COUNT(*) FROM knowledge_entities GROUP BY entity_type")
        st.sidebar.markdown("---")
        st.sidebar.markdown("**KG Stats**")
        for row in cur.fetchall():
            st.sidebar.markdown(f"- **{row[0]}**: {row[1]}")
        conn.close()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Ingestion / Sync Administration**")
    if st.sidebar.button("Sync Today's Launches", use_container_width=True):
        with st.sidebar.status("Syncing & Rebuilding Graph...") as status:
            try:
                sync_and_rebuild_today_launches()
                status.update(label="Sync Completed!", state="complete")
                
                # Clear session state keys to force reload of products/KG summaries
                st.session_state.chat_history = []
                st.session_state.left_response_id = None
                st.session_state.right_response_id = None
                if "all_products" in st.session_state:
                    del st.session_state.all_products
                if "kg_summary" in st.session_state:
                    del st.session_state.kg_summary
                if "today_products" in st.session_state:
                    del st.session_state.today_products
                st.toast("Today's products synchronized successfully! Cache refreshed.")
                st.rerun()
            except Exception as e:
                status.update(label=f"Sync Failed: {e}", state="error")

    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Chat & Refresh Cache", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.left_response_id = None
        st.session_state.right_response_id = None
        if "all_products" in st.session_state:
            del st.session_state.all_products
        if "kg_summary" in st.session_state:
            del st.session_state.kg_summary
        st.rerun()


# ─────────────────────────────────────────────────────────────
# App UI
# ─────────────────────────────────────────────────────────────

st.title("Product Hunt AI Assistant (Milestone 3 Agent)")
st.markdown(
    "Ask about products, features, pricing, competitors, **market gaps**, "
    "**opportunity detection**, and get grounded explainability from the Knowledge Graph."
)

render_sidebar()

# CSS for a vertical divider between columns
st.markdown("""
<style>
div[data-testid="column"]:nth-of-type(1) {
    border-right: 2px solid #2d3a4f;
    padding-right: 20px;
}
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "left_response_id" not in st.session_state:
    st.session_state.left_response_id = None
if "right_response_id" not in st.session_state:
    st.session_state.right_response_id = None

if "today_products" not in st.session_state:
    st.session_state.today_products = get_today_products()
if "all_products" not in st.session_state:
    st.session_state.all_products = get_all_db_products()
if "kg_summary" not in st.session_state:
    with st.spinner("Loading Knowledge Graph summary..."):
        st.session_state.kg_summary = get_kg_summary_context()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Left Bot (With KG)")
with col2:
    st.subheader("Right Bot (Without KG)")

for chat in st.session_state.chat_history:
    c1, c2 = st.columns(2)
    with c1:
        with st.chat_message("user"):
            st.markdown(chat["user"])
        with st.chat_message("assistant"):
            st.markdown(chat["left"])
    with c2:
        with st.chat_message("user"):
            st.markdown(chat["user"])
        with st.chat_message("assistant"):
            st.markdown(chat["right"])




st.markdown("---")
st.markdown("**Suggested queries to try:**")
st.markdown(
    "- What are the top KG insights for today's Product Hunt launches?  \n"
    "- Which products appear uncontested in the KG?  \n"
    "- Summarize KG stats and market gap opportunities.  \n"
    "- How does today's launch compare to the historical KG?"
)

if prompt := st.chat_input("E.g. What do you know about Mora Marketer?"):
    # Extract session state variables in main thread to avoid thread-context issues
    today_products_main = st.session_state.today_products
    kg_summary_main = st.session_state.kg_summary
    left_response_id_main = st.session_state.left_response_id
    all_products_main = st.session_state.all_products
    right_response_id_main = st.session_state.right_response_id

    async def get_both_responses():
        left_res, left_id = await generate_kg_response(
            prompt,
            today_products_main,
            kg_summary_main,
            left_response_id_main
        )
        right_res, right_id = await generate_baseline_response(
            prompt,
            today_products_main,
            all_products_main,
            right_response_id_main
        )
        return left_res, left_id, right_res, right_id

    # Show loading indicator and fetch responses simultaneously using the thread runner
    with st.spinner("Both bots are generating responses..."):
        left_response, new_left_id, right_response, new_right_id = run_async_in_thread(get_both_responses())

    st.session_state.left_response_id = new_left_id
    st.session_state.right_response_id = new_right_id

    st.session_state.chat_history.append({
        "user": prompt,
        "left": left_response,
        "right": right_response
    })
    st.rerun()


