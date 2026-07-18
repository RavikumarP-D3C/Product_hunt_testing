"""
kg_tools.py — Knowledge Graph Query Tools for Milestone 3

Three focused tools that expose the full SQLite knowledge graph to the chatbot:
  1. query_kg_for_product()          - Full edges for a product (features, targets, competes, belongs)
  2. find_market_gaps()              - Unserved customer segments & missing features in a category
  3. find_uncontested_products()     - Today's products that have no KG competitors
  4. cross_reference_today_vs_kg()  - Properly matches today's cache products to KG categories
                                       WITHOUT mixing up historical KG products as today's launches
"""

import os
import json
import sqlite3
from typing import Any, Dict, List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_graph.db")


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _fuzzy_find_product(name: str, cur: sqlite3.Cursor) -> Optional[str]:
    """
    Returns the SQLite entity id for the best-matching Product.
    Tries exact match first, then LIKE partial match.
    """
    # 1. Exact
    cur.execute(
        "SELECT id FROM knowledge_entities WHERE entity_type='Product' AND LOWER(canonical_name)=LOWER(?)",
        (name,)
    )
    row = cur.fetchone()
    if row:
        return row["id"]

    # 2. Partial (starts with or contains)
    cur.execute(
        "SELECT id, canonical_name FROM knowledge_entities WHERE entity_type='Product' AND LOWER(canonical_name) LIKE LOWER(?)",
        (f"%{name}%",)
    )
    row = cur.fetchone()
    if row:
        return row["id"]

    return None


# ─────────────────────────────────────────────────────────────
# Tool 1 — Full product neighbourhood query
# ─────────────────────────────────────────────────────────────

def query_kg_for_product(product_name: str) -> Dict[str, Any]:
    """
    Returns every KG edge touching the given product:
      - features (HAS_FEATURE)
      - target audiences (TARGETS)
      - categories (BELONGS_TO)
      - competitors (COMPETES_WITH) with evidence_text + confidence
      - similar products (SIMILAR_TO) with strength score
      - developed by company (DEVELOPS, reversed)

    Returns a dict ready to be serialised to JSON and passed to the LLM.
    """
    if not os.path.exists(DB_PATH):
        return {"error": "knowledge_graph.db not found"}

    try:
        conn = _get_conn()
        cur = conn.cursor()

        entity_id = _fuzzy_find_product(product_name, cur)
        if not entity_id:
            return {"error": f"Product '{product_name}' not found in the knowledge graph."}

        # Canonical name
        cur.execute("SELECT canonical_name FROM knowledge_entities WHERE id=?", (entity_id,))
        canonical = cur.fetchone()["canonical_name"]

        result: Dict[str, Any] = {
            "product": canonical,
            "features": [],
            "target_audience": [],
            "categories": [],
            "competes_with": [],
            "similar_to": [],
            "developed_by": None
        }

        # Features
        cur.execute("""
            SELECT e2.canonical_name
            FROM knowledge_relationships r
            JOIN knowledge_entities e2 ON r.target_entity_id = e2.id
            WHERE r.source_entity_id=? AND r.relationship_type='HAS_FEATURE' AND r.status='active'
        """, (entity_id,))
        result["features"] = [row["canonical_name"] for row in cur.fetchall()]

        # Target Audience
        cur.execute("""
            SELECT e2.canonical_name
            FROM knowledge_relationships r
            JOIN knowledge_entities e2 ON r.target_entity_id = e2.id
            WHERE r.source_entity_id=? AND r.relationship_type='TARGETS' AND r.status='active'
        """, (entity_id,))
        result["target_audience"] = [row["canonical_name"] for row in cur.fetchall()]

        # Categories
        cur.execute("""
            SELECT e2.canonical_name
            FROM knowledge_relationships r
            JOIN knowledge_entities e2 ON r.target_entity_id = e2.id
            WHERE r.source_entity_id=? AND r.relationship_type='BELONGS_TO' AND r.status='active'
        """, (entity_id,))
        result["categories"] = [row["canonical_name"] for row in cur.fetchall()]

        # Competes With (outgoing + incoming)
        cur.execute("""
            SELECT e2.canonical_name, r.evidence_confidence, r.relationship_strength,
                   r.evidence_text, r.inference_type
            FROM knowledge_relationships r
            JOIN knowledge_entities e2 ON r.target_entity_id = e2.id
            WHERE r.source_entity_id=? AND r.relationship_type='COMPETES_WITH' AND r.status='active'
        """, (entity_id,))
        for row in cur.fetchall():
            result["competes_with"].append({
                "product": row["canonical_name"],
                "confidence": row["evidence_confidence"],
                "strength": round(row["relationship_strength"], 3) if row["relationship_strength"] else None,
                "evidence": row["evidence_text"],
                "inference_type": row["inference_type"]
            })

        # Reverse COMPETES_WITH (others that list this product as competitor)
        cur.execute("""
            SELECT e1.canonical_name, r.evidence_confidence, r.relationship_strength, r.evidence_text
            FROM knowledge_relationships r
            JOIN knowledge_entities e1 ON r.source_entity_id = e1.id
            WHERE r.target_entity_id=? AND r.relationship_type='COMPETES_WITH' AND r.status='active'
        """, (entity_id,))
        seen = {c["product"] for c in result["competes_with"]}
        for row in cur.fetchall():
            if row["canonical_name"] not in seen:
                result["competes_with"].append({
                    "product": row["canonical_name"],
                    "confidence": row["evidence_confidence"],
                    "strength": round(row["relationship_strength"], 3) if row["relationship_strength"] else None,
                    "evidence": row["evidence_text"],
                    "inference_type": "derived (reverse edge)"
                })

        # Similar To (top 10 by strength)
        cur.execute("""
            SELECT e2.canonical_name, r.relationship_strength, r.evidence_text
            FROM knowledge_relationships r
            JOIN knowledge_entities e2 ON r.target_entity_id = e2.id
            WHERE r.source_entity_id=? AND r.relationship_type='SIMILAR_TO' AND r.status='active'
            ORDER BY r.relationship_strength DESC LIMIT 10
        """, (entity_id,))
        for row in cur.fetchall():
            result["similar_to"].append({
                "product": row["canonical_name"],
                "strength": round(row["relationship_strength"], 3) if row["relationship_strength"] else None,
                "evidence": row["evidence_text"]
            })

        # Developed by Company (reverse DEVELOPS edge)
        cur.execute("""
            SELECT e1.canonical_name
            FROM knowledge_relationships r
            JOIN knowledge_entities e1 ON r.source_entity_id = e1.id
            WHERE r.target_entity_id=? AND r.relationship_type='DEVELOPS' AND r.status='active'
        """, (entity_id,))
        dev_row = cur.fetchone()
        if dev_row:
            result["developed_by"] = dev_row["canonical_name"]

        conn.close()
        return result

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────
# Tool 2 — Market Gap Analysis for a category
# ─────────────────────────────────────────────────────────────

def find_market_gaps(category: str) -> Dict[str, Any]:
    """
    For a given product category (e.g. 'Social Media Tools', 'Developer Tools'):
    1. Find all products that BELONG_TO that category (fuzzy match)
    2. Collect ALL features across those products
    3. Collect ALL customer segments targeted
    4. Cross-reference: which segments are targeted by FEWER than half the products?
       Those are underserved segments (potential gaps).
    5. Return top features, underserved segments, and full product list.
    """
    if not os.path.exists(DB_PATH):
        return {"error": "knowledge_graph.db not found"}

    try:
        conn = _get_conn()
        cur = conn.cursor()

        # 1. Find matching categories (fuzzy)
        cur.execute(
            "SELECT id, canonical_name FROM knowledge_entities WHERE entity_type='Category' AND LOWER(canonical_name) LIKE LOWER(?)",
            (f"%{category}%",)
        )
        cat_rows = cur.fetchall()
        if not cat_rows:
            return {"error": f"No category matching '{category}' found in knowledge graph."}

        matched_categories = [r["canonical_name"] for r in cat_rows]
        cat_ids = [r["id"] for r in cat_rows]

        # 2. Find products in those categories
        placeholders = ",".join("?" * len(cat_ids))
        cur.execute(f"""
            SELECT DISTINCT e1.id, e1.canonical_name
            FROM knowledge_relationships r
            JOIN knowledge_entities e1 ON r.source_entity_id = e1.id
            WHERE r.target_entity_id IN ({placeholders}) AND r.relationship_type='BELONGS_TO' AND r.status='active'
        """, cat_ids)
        products = cur.fetchall()

        if not products:
            return {
                "matched_categories": matched_categories,
                "error": "No products found in this category in the knowledge graph."
            }

        product_ids = [p["id"] for p in products]
        product_names = [p["canonical_name"] for p in products]
        total_products = len(product_ids)

        # 3. Features across all products
        ph = ",".join("?" * len(product_ids))
        cur.execute(f"""
            SELECT e2.canonical_name, COUNT(DISTINCT r.source_entity_id) as product_count
            FROM knowledge_relationships r
            JOIN knowledge_entities e2 ON r.target_entity_id = e2.id
            WHERE r.source_entity_id IN ({ph}) AND r.relationship_type='HAS_FEATURE' AND r.status='active'
            GROUP BY e2.canonical_name
            ORDER BY product_count DESC
        """, product_ids)
        all_features = [{"feature": r["canonical_name"], "covered_by_n_products": r["product_count"]} for r in cur.fetchall()]

        # 4. Customer segments & coverage
        cur.execute(f"""
            SELECT e2.canonical_name, COUNT(DISTINCT r.source_entity_id) as product_count
            FROM knowledge_relationships r
            JOIN knowledge_entities e2 ON r.target_entity_id = e2.id
            WHERE r.source_entity_id IN ({ph}) AND r.relationship_type='TARGETS' AND r.status='active'
            GROUP BY e2.canonical_name
            ORDER BY product_count ASC
        """, product_ids)
        all_segments = [{"segment": r["canonical_name"], "covered_by_n_products": r["product_count"]} for r in cur.fetchall()]

        # 5. Underserved segments = targeted by < half the products
        underserved = [s for s in all_segments if s["covered_by_n_products"] < max(1, total_products // 2)]

        conn.close()
        return {
            "matched_categories": matched_categories,
            "total_products_in_category": total_products,
            "products": product_names,
            "all_features": all_features[:30],        # top 30 most common
            "all_customer_segments": all_segments[:30],
            "underserved_segments": underserved[:15],  # potential gaps
        }

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────
# Tool 3 — Find uncontested products from today's cache
# ─────────────────────────────────────────────────────────────

def find_uncontested_products(today_products: List[Dict]) -> Dict[str, Any]:
    """
    For each product in today_products list, check if it (or anything similar)
    exists in the knowledge graph with a COMPETES_WITH or SIMILAR_TO edge.
    Products with NO such match = uncontested / novel niche.

    Returns:
        - uncontested: products with zero KG competitors
        - partially_contested: products with some similarity but no direct competitors
        - well_contested: products that directly compete with KG products
    """
    if not os.path.exists(DB_PATH):
        return {"error": "knowledge_graph.db not found"}

    try:
        conn = _get_conn()
        cur = conn.cursor()

        uncontested = []
        partially_contested = []
        well_contested = []

        for prod in today_products:
            name = prod.get("name", "")
            tagline = prod.get("tagline", "")
            if not name:
                continue

            entity_id = _fuzzy_find_product(name, cur)

            if not entity_id:
                # Product not in KG at all → totally new
                uncontested.append({
                    "product": name,
                    "tagline": tagline,
                    "reason": "Not found in knowledge graph at all — completely new entry"
                })
                continue

            # Check COMPETES_WITH
            cur.execute("""
                SELECT COUNT(*) as cnt FROM knowledge_relationships
                WHERE (source_entity_id=? OR target_entity_id=?)
                AND relationship_type='COMPETES_WITH' AND status='active'
            """, (entity_id, entity_id))
            comp_count = cur.fetchone()["cnt"]

            # Check SIMILAR_TO
            cur.execute("""
                SELECT COUNT(*) as cnt FROM knowledge_relationships
                WHERE (source_entity_id=? OR target_entity_id=?)
                AND relationship_type='SIMILAR_TO' AND status='active'
            """, (entity_id, entity_id))
            sim_count = cur.fetchone()["cnt"]

            if comp_count == 0 and sim_count == 0:
                uncontested.append({
                    "product": name,
                    "tagline": tagline,
                    "reason": "In KG but has zero COMPETES_WITH or SIMILAR_TO edges"
                })
            elif comp_count == 0 and sim_count > 0:
                partially_contested.append({
                    "product": name,
                    "tagline": tagline,
                    "similar_count": sim_count,
                    "direct_competitors": 0
                })
            else:
                well_contested.append({
                    "product": name,
                    "tagline": tagline,
                    "competitor_count": comp_count,
                    "similar_count": sim_count
                })

        conn.close()
        return {
            "uncontested": uncontested,
            "partially_contested": partially_contested,
            "well_contested": well_contested,
            "summary": {
                "total_today": len(today_products),
                "uncontested_count": len(uncontested),
                "partially_contested_count": len(partially_contested),
                "well_contested_count": len(well_contested)
            }
        }

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────
# Tool 4 — Cross-reference today's products vs KG categories
# (Fixes Bug 1: LLM mixing historical KG products as today's launches)
# ─────────────────────────────────────────────────────────────

def cross_reference_today_vs_kg(today_products: List[Dict], target_category: Optional[str] = None) -> Dict[str, Any]:
    """
    Correctly cross-references today's Product Hunt cache against the KG.

    For each of today's products:
      - Checks if it exists in KG by name (fuzzy match)
      - If YES: fetches its categories, features, and target audience from KG
      - If NO: marks it as new and tries to infer its KG category from its tagline keywords

    If target_category is provided, it also fetches that category's market gaps
    and checks if any of today's products match it.

    KEY: This function NEVER returns historical KG products as "today's launches."
    It only returns products that were in the today_products input list.
    """
    if not os.path.exists(DB_PATH):
        return {"error": "knowledge_graph.db not found"}

    try:
        conn = _get_conn()
        cur = conn.cursor()

        # Load all known KG categories
        cur.execute("SELECT canonical_name FROM knowledge_entities WHERE entity_type='Category'")
        all_kg_categories = [row["canonical_name"] for row in cur.fetchall()]

        def infer_category_from_tagline(tagline: str, name: str) -> List[str]:
            """Simple keyword matching to guess KG category from product tagline."""
            text = (tagline + " " + name).lower()
            matches = []
            for cat in all_kg_categories:
                cat_words = set(cat.lower().split())
                text_words = set(text.split())
                if cat_words & text_words:
                    matches.append(cat)
            return matches[:3]

        results = {
            "data_source_note": (
                "IMPORTANT: Only today's products are listed here. "
                "Historical KG product names do NOT appear in this list. "
                "KG data (features/segments) is pulled for context only."
            ),
            "today_products": [],
            "category_gap_analysis": None
        }

        for prod in today_products:
            name    = prod.get("name", "")
            tagline = prod.get("tagline", "")
            website = prod.get("website", "")

            if not name:
                continue

            entry: Dict[str, Any] = {
                "name":    name,
                "tagline": tagline,
                "website": website,
                "in_kg":   False,
                "kg_data": None,
                "inferred_kg_categories": []
            }

            # Check if this today's product exists in KG
            entity_id = _fuzzy_find_product(name, cur)
            if entity_id:
                entry["in_kg"] = True
                # Pull its KG neighbourhood
                cur.execute("""
                    SELECT e2.canonical_name FROM knowledge_relationships r
                    JOIN knowledge_entities e2 ON r.target_entity_id = e2.id
                    WHERE r.source_entity_id=? AND r.relationship_type='BELONGS_TO' AND r.status='active'
                """, (entity_id,))
                kg_cats = [r["canonical_name"] for r in cur.fetchall()]

                cur.execute("""
                    SELECT e2.canonical_name FROM knowledge_relationships r
                    JOIN knowledge_entities e2 ON r.target_entity_id = e2.id
                    WHERE r.source_entity_id=? AND r.relationship_type='HAS_FEATURE' AND r.status='active'
                """, (entity_id,))
                kg_feats = [r["canonical_name"] for r in cur.fetchall()]

                cur.execute("""
                    SELECT e2.canonical_name FROM knowledge_relationships r
                    JOIN knowledge_entities e2 ON r.target_entity_id = e2.id
                    WHERE r.source_entity_id=? AND r.relationship_type='TARGETS' AND r.status='active'
                """, (entity_id,))
                kg_segs = [r["canonical_name"] for r in cur.fetchall()]

                cur.execute("""
                    SELECT COUNT(*) as cnt FROM knowledge_relationships
                    WHERE (source_entity_id=? OR target_entity_id=?)
                    AND relationship_type='COMPETES_WITH' AND status='active'
                """, (entity_id, entity_id))
                comp_count = cur.fetchone()["cnt"]

                entry["kg_data"] = {
                    "categories":    kg_cats,
                    "features":      kg_feats,
                    "target_audience": kg_segs,
                    "has_competitors": comp_count > 0,
                    "competitor_count": comp_count
                }
            else:
                # Product is NEW — infer KG category from tagline
                entry["inferred_kg_categories"] = infer_category_from_tagline(tagline, name)

            results["today_products"].append(entry)

        # Optional: pull category gap analysis for context
        if target_category:
            results["category_gap_analysis"] = find_market_gaps(target_category)

        conn.close()
        return results

    except Exception as e:
        return {"error": str(e)}

