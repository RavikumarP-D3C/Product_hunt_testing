import os
import time
import random
import logging
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Retry configuration for 429 rate-limit errors
_MAX_RETRIES = 5          # maximum number of retry attempts
_BASE_BACKOFF_S = 2.0     # initial backoff in seconds
_MAX_BACKOFF_S = 64.0     # cap on backoff duration
_JITTER_FACTOR = 0.25     # ±25% random jitter to avoid thundering-herd

# GraphQL query to fetch posts from Product Hunt v2 API
PRODUCT_HUNT_GRAPHQL_QUERY = """
query GetLatestPosts($limit: Int!, $cursor: String) {
  posts(first: $limit, after: $cursor) {
    edges {
      node {
        id
        name
        tagline
        description
        website
        votesCount
        commentsCount
        createdAt
        topics {
          edges {
            node {
              name
            }
          }
        }
        makers {
          id
          name
        }
      }
      cursor
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""

def fetch_latest_products(limit: int = 20, cursor: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch newly launched products from Product Hunt.
    If PRODUCT_HUNT_DEVELOPER_TOKEN is not configured, falls back to generating mock data.
    
    Args:
        limit (int): Number of products to fetch. Default is 20.
        cursor (str): Pagination cursor.
        
    Returns:
        List[Dict[str, Any]]: A list of products matching the output schema.
    """
    token = os.getenv("PRODUCTHUNT_DEVELOPER_TOKEN") or os.getenv("PRODUCT_HUNT_DEVELOPER_TOKEN")
    
    if not token:
        logger.warning("PRODUCTHUNT_DEVELOPER_TOKEN environment variable not set. Falling back to mock data.")
        return _generate_mock_products(limit)

    url = "https://api.producthunt.com/v2/api/graphql"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    variables = {
        "limit": limit,
        "cursor": cursor
    }
    
    payload = {
        "query": PRODUCT_HUNT_GRAPHQL_QUERY,
        "variables": variables
    }
    
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=15)

            # ── 429: rate limited ────────────────────────────────────────────
            if response.status_code == 429:
                if attempt == _MAX_RETRIES:
                    logger.error(
                        f"Product Hunt API returned 429 after {_MAX_RETRIES} retries — giving up."
                    )
                    response.raise_for_status()  # propagate to caller

                # Honour Retry-After if the server tells us how long to wait
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = None
                else:
                    wait = None

                if wait is None:
                    # Exponential backoff: 2s, 4s, 8s … capped at 64s
                    base = min(_BASE_BACKOFF_S * (2 ** attempt), _MAX_BACKOFF_S)
                    # ±25% jitter so simultaneous timers don't all retry together
                    jitter = base * _JITTER_FACTOR * (2 * random.random() - 1)
                    wait = base + jitter

                logger.warning(
                    f"Product Hunt API rate-limited (429). "
                    f"Attempt {attempt + 1}/{_MAX_RETRIES}. "
                    f"Retrying in {wait:.1f}s…"
                )
                time.sleep(wait)
                continue  # retry the request

            # ── Any other HTTP error: fail immediately ────────────────────────
            response.raise_for_status()

            result = response.json()
            if "errors" in result:
                errors = result["errors"]
                logger.error(f"Product Hunt API returned errors: {errors}")
                raise ValueError(f"Product Hunt GraphQL Error: {errors[0].get('message')}")

            posts_data = result.get("data", {}).get("posts", {})
            edges = posts_data.get("edges", [])

            products = []
            for edge in edges:
                node = edge.get("node", {})
                product = {
                    "id": str(node.get("id", "")),
                    "name": str(node.get("name", "")),
                    "tagline": str(node.get("tagline", "")),
                    "description": str(node.get("description", "")),
                    "website": str(node.get("website", "")),
                    "topics": [
                        t.get("node", {}).get("name", "")
                        for t in (node.get("topics") or {}).get("edges") or []
                        if t
                    ],
                    "votes": int(node.get("votesCount", 0)),
                    "comments": int(node.get("commentsCount", 0)),
                    "launch_date": str(node.get("createdAt", "")),
                    "makers": [
                        {"id": str(m.get("id", "")), "name": str(m.get("name", ""))}
                        for m in node.get("makers", []) if m
                    ]
                }
                products.append(product)

            return products

        except requests.exceptions.HTTPError:
            # Already logged above for 429; for everything else, surface immediately
            raise
        except Exception as e:
            logger.error(f"Error fetching products from Product Hunt: {e}", exc_info=True)
            raise

    # Should never reach here, but satisfy type-checker
    return []

def _generate_mock_products(limit: int) -> List[Dict[str, Any]]:
    """
    Helper function to generate fallback rich mock data representing newly launched products.
    """
    import datetime
    
    mock_source = [
        {
            "id": "10001",
            "name": "InfluenzeAI Agent",
            "tagline": "Autonomous marketing agents for high-converting social media posts",
            "description": "Generate, edit, schedule and distribute visual content on LinkedIn, Instagram and Facebook using multi-agent orchestration.",
            "website": "https://influenzeai.io",
            "topics": ["Artificial Intelligence", "Marketing", "SaaS"],
            "votes": 345,
            "comments": 42,
            "makers": [{"id": "99", "name": "DeepMind Team"}]
        },
        {
            "id": "10002",
            "name": "Antigravity Devtool",
            "tagline": "AI pair programmer designed by Google DeepMind developers",
            "description": "Accelerate software delivery with a sandbox terminal, browser automation, and multi-file editing support.",
            "website": "https://antigravity.dev",
            "topics": ["Developer Tools", "AI", "Productivity"],
            "votes": 820,
            "comments": 105,
            "makers": [{"id": "101", "name": "Lily the Coding Assistant"}]
        },
        {
            "id": "10003",
            "name": "QuickBriefing",
            "tagline": "Summarize everything happening in your industry in 30 seconds",
            "description": "Get daily briefings containing trending news, product launches and relevant insights directly in your email.",
            "website": "https://quickbriefing.com",
            "topics": ["News", "AI", "Marketing"],
            "votes": 120,
            "comments": 15,
            "makers": [{"id": "105", "name": "John Doe"}]
        }
    ]
    
    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    products = []
    for idx, item in enumerate(mock_source[:limit]):
        product = item.copy()
        product["launch_date"] = now_str
        products.append(product)
        
    return products
