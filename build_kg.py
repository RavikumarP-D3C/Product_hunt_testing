import os
import re
import sqlite3
import uuid
import json
from datetime import datetime, timezone
from urllib.parse import urlparse
from pymongo import MongoClient
from dotenv import load_dotenv

# Define paths and load environment variables from Backend/.env
LOCAL_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(LOCAL_ENV):
    ENV_PATH = LOCAL_ENV
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ENV_PATH = os.path.join(BASE_DIR, "Backend", ".env")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_graph.db")

print(f"Loading environment variables from: {ENV_PATH}")
load_dotenv(dotenv_path=ENV_PATH)

def get_mongo_connection():
    mongo_uri = os.getenv("MONGO_CONNECTION_STRING")
    if not mongo_uri:
        raise ValueError("MONGO_CONNECTION_STRING is not set in environment variables.")
    client = MongoClient(mongo_uri)
    return client["influenze_ai_marketing"]

def init_db(conn):
    """
    Creates SQL tables for the knowledge graph.
    """
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. Canonical Entity Registry
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS knowledge_entities (
        id TEXT PRIMARY KEY,
        entity_type TEXT NOT NULL,
        canonical_name TEXT NOT NULL,
        source_table TEXT,
        source_record_id TEXT,
        canonicalization_status TEXT NOT NULL,
        attributes_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (entity_type, source_table, source_record_id)
    )""")

    # 2. Typed Relationships
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS knowledge_relationships (
        id TEXT PRIMARY KEY,
        source_entity_id TEXT NOT NULL REFERENCES knowledge_entities(id),
        relationship_type TEXT NOT NULL,
        target_entity_id TEXT NOT NULL REFERENCES knowledge_entities(id),
        attributes_json TEXT NOT NULL DEFAULT '{}',
        evidence_confidence REAL,
        relationship_strength REAL,
        inference_type TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active',
        source_record_id TEXT,
        source_url TEXT,
        evidence_text TEXT,
        observed_at TEXT,
        valid_from TEXT,
        valid_to TEXT,
        generation_method TEXT,
        model_version TEXT,
        rule_version TEXT,
        schema_version TEXT,
        pipeline_run_id TEXT
    )""")

    # 3. Observations History
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS knowledge_relationship_observations (
        id TEXT PRIMARY KEY,
        relationship_id TEXT NOT NULL REFERENCES knowledge_relationships(id),
        observed_at TEXT NOT NULL,
        attributes_json TEXT NOT NULL DEFAULT '{}',
        evidence_json TEXT NOT NULL DEFAULT '[]',
        evidence_confidence REAL,
        relationship_strength REAL,
        generation_method TEXT,
        change_reason TEXT,
        created_at TEXT NOT NULL
    )""")

    # 4. Resolution Candidates
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS entity_resolution_candidates (
        id TEXT PRIMARY KEY,
        provisional_entity_id TEXT NOT NULL REFERENCES knowledge_entities(id),
        candidate_entity_id TEXT NOT NULL REFERENCES knowledge_entities(id),
        match_score REAL,
        match_factors_json TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL,
        resolution_reason TEXT,
        reviewed_by TEXT,
        reviewed_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""")

    # Indexes for quick traversals
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_source_entity ON knowledge_relationships (source_entity_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_target_entity ON knowledge_relationships (target_entity_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_relationship_type ON knowledge_relationships (relationship_type)")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_kg_active_relationships 
        ON knowledge_relationships (source_entity_id, relationship_type, target_entity_id)
        WHERE status = 'active' AND valid_to IS NULL
    """)
    
    conn.commit()
    print("[INIT] Database schemas and indexes verified.")

def clean_domain(url):
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        domain = domain.replace("www.", "").lower().strip()
        return domain
    except Exception:
        return ""

def clean_text_id(text):
    if not text:
        return ""
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower().strip()

def main():
    print("=== STARTING REAL DATA KNOWLEDGE GRAPH INGESTION ===")
    
    # 1. Initialize SQLite Database
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    cursor = conn.cursor()

    # 2. Connect to MongoDB
    try:
        db = get_mongo_connection()
    except Exception as e:
        print(f"[ERROR] Failed to connect to MongoDB: {e}")
        return

    now = datetime.now(timezone.utc).isoformat()
    
    # Mappings to keep track of SQLite entity IDs
    company_mapping = {}  # company_id -> SQLite UUID
    product_mapping = {}  # product_hunt_id -> SQLite UUID
    other_entities = {}   # (type, name) -> SQLite UUID
    
    # 3. Load and ingest Companies
    print("\n[INGEST] Ingesting companies from MongoDB...")
    companies = list(db.companies.find())
    print(f"Found {len(companies)} companies in MongoDB.")
    
    for co in companies:
        co_id = co.get("company_id")
        name = co.get("name")
        website = co.get("website", "")
        domain = clean_domain(website)
        
        intel = co.get("intelligence", {})
        overview = intel.get("company_overview", "")
        industry = co.get("industry", "")
        
        attributes = {
            "domain": domain,
            "website": website,
            "industry": industry,
            "overview": overview
        }
        
        # Ingest company
        ent_id = str(uuid.uuid4())
        try:
            cursor.execute("""
                INSERT INTO knowledge_entities 
                (id, entity_type, canonical_name, source_table, source_record_id, canonicalization_status, attributes_json, created_at, updated_at)
                VALUES (?, 'Company', ?, 'companies', ?, 'confirmed', ?, ?, ?)
            """, (ent_id, name, str(co.get("_id")), json.dumps(attributes), now, now))
            company_mapping[co_id] = ent_id
        except sqlite3.IntegrityError:
            # Already exists in the database
            cursor.execute("SELECT id FROM knowledge_entities WHERE entity_type = 'Company' AND source_record_id = ?", (str(co.get("_id")),))
            row = cursor.fetchone()
            if row:
                company_mapping[co_id] = row[0]

    print(f"Successfully registered {len(company_mapping)} companies in SQLite.")

    # 4. Load and ingest Products
    print("\n[INGEST] Ingesting product intelligence from MongoDB...")
    products = list(db.product_intelligence.find())
    print(f"Found {len(products)} products in MongoDB.")
    
    product_attributes_lookup = {} # SQLite UUID -> dict of categories, customer segments, features

    # --- PRE-PASS: resolve orphan product company_ids by website domain matching ---
    # Many products have company_id like 'co_ph_XXXXXX' that don't exist in the companies collection.
    # Try to match them to a company by comparing website domains.
    company_domain_map = {}  # domain string -> company_id
    for co in companies:
        co_domain = clean_domain(co.get("website", ""))
        if co_domain:
            company_domain_map[co_domain] = co.get("company_id")

    resolved_count = 0
    for prod in products:
        if prod.get("company_id", "").startswith("co_ph_"):
            prod_domain = clean_domain(prod.get("website", ""))
            if prod_domain and prod_domain in company_domain_map:
                prod["company_id"] = company_domain_map[prod_domain]
                resolved_count += 1
    print(f"[RESOLVE] Resolved {resolved_count} orphan product<->company links via domain matching.")

    for prod in products:
        ph_id = prod.get("product_hunt_id")
        name = prod.get("name")
        co_id = prod.get("company_id")
        website = prod.get("website", "")
        tagline = prod.get("tagline", "")
        categories = prod.get("categories", [])
        votes = prod.get("votes", 0)
        ph_url = prod.get("product_hunt_url", "")
        
        attributes = {
            "tagline": tagline,
            "website": website,
            "votes": votes,
            "product_hunt_url": ph_url
        }
        
        # Register product
        ent_id = str(uuid.uuid4())
        try:
            cursor.execute("""
                INSERT INTO knowledge_entities 
                (id, entity_type, canonical_name, source_table, source_record_id, canonicalization_status, attributes_json, created_at, updated_at)
                VALUES (?, 'Product', ?, 'product_intelligence', ?, 'confirmed', ?, ?, ?)
            """, (ent_id, name, str(prod.get("_id")), json.dumps(attributes), now, now))
            product_mapping[ph_id] = ent_id
        except sqlite3.IntegrityError:
            cursor.execute("SELECT id FROM knowledge_entities WHERE entity_type = 'Product' AND source_record_id = ?", (str(prod.get("_id")),))
            row = cursor.fetchone()
            if row:
                ent_id = row[0]
                product_mapping[ph_id] = ent_id
        
        product_attributes_lookup[ent_id] = {
            "categories": [c.strip() for c in categories if c],
            "customer_segments": [],
            "features": [],
            "name": name,
            "company_id": co_id
        }

        # 5. Ingest Categories and link Product BELONGS_TO Category
        for cat in categories:
            if not cat:
                continue
            cat_key = ("Category", cat.strip())
            if cat_key not in other_entities:
                cat_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT OR IGNORE INTO knowledge_entities 
                    (id, entity_type, canonical_name, source_table, source_record_id, canonicalization_status, attributes_json, created_at, updated_at)
                    VALUES (?, 'Category', ?, 'product_intelligence', NULL, 'confirmed', '{}', ?, ?)
                """, (cat_id, cat.strip(), now, now))
                
                # Fetch ID in case it was ignored
                cursor.execute("SELECT id FROM knowledge_entities WHERE entity_type = 'Category' AND canonical_name = ?", (cat.strip(),))
                row = cursor.fetchone()
                other_entities[cat_key] = row[0] if row else cat_id
            
            cat_ent_id = other_entities[cat_key]
            
            # Create relationship
            rel_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT OR IGNORE INTO knowledge_relationships 
                (id, source_entity_id, relationship_type, target_entity_id, evidence_confidence, relationship_strength, inference_type, status, evidence_text, valid_from)
                VALUES (?, ?, 'BELONGS_TO', ?, 1.0, 1.0, 'extracted', 'active', 'Extracted from Product Hunt categories.', ?)
            """, (rel_id, ent_id, cat_ent_id, now))

    # 6. Ingest Company-linked features and customer segments
    print("\n[INGEST] Linking products to companies and extracting customer segments/features...")
    for co in companies:
        co_id = co.get("company_id")
        co_ent_id = company_mapping.get(co_id)
        if not co_ent_id:
            continue
            
        intel = co.get("intelligence", {})
        target_audience = intel.get("target_audience", [])
        key_features = intel.get("key_features", [])
        
        # Find all products developed by this company
        for ph_id, prod_ent_id in product_mapping.items():
            prod_lookup = product_attributes_lookup.get(prod_ent_id)
            if prod_lookup and prod_lookup["company_id"] == co_id:
                # Add DEVELOPED_BY relationship
                rel_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT OR IGNORE INTO knowledge_relationships 
                    (id, source_entity_id, relationship_type, target_entity_id, evidence_confidence, relationship_strength, inference_type, status, evidence_text, valid_from)
                    VALUES (?, ?, 'DEVELOPS', ?, 1.0, 1.0, 'extracted', 'active', 'Determined from company mapping.', ?)
                """, (rel_id, co_ent_id, prod_ent_id, now))
                
                # Link Features
                for feat in key_features:
                    if not feat:
                        continue
                    feat_key = ("Feature", feat.strip())
                    if feat_key not in other_entities:
                        feat_id = str(uuid.uuid4())
                        cursor.execute("""
                            INSERT OR IGNORE INTO knowledge_entities 
                            (id, entity_type, canonical_name, source_table, source_record_id, canonicalization_status, attributes_json, created_at, updated_at)
                            VALUES (?, 'Feature', ?, 'companies', NULL, 'confirmed', '{}', ?, ?)
                        """, (feat_id, feat.strip(), now, now))
                        cursor.execute("SELECT id FROM knowledge_entities WHERE entity_type = 'Feature' AND canonical_name = ?", (feat.strip(),))
                        row = cursor.fetchone()
                        other_entities[feat_key] = row[0] if row else feat_id
                    
                    feat_ent_id = other_entities[feat_key]
                    prod_lookup["features"].append(feat.strip())
                    
                    rel_id = str(uuid.uuid4())
                    cursor.execute("""
                        INSERT OR IGNORE INTO knowledge_relationships 
                        (id, source_entity_id, relationship_type, target_entity_id, evidence_confidence, relationship_strength, inference_type, status, evidence_text, valid_from)
                        VALUES (?, ?, 'HAS_FEATURE', ?, 1.0, 1.0, 'extracted', 'active', 'Extracted from company key features.', ?)
                    """, (rel_id, prod_ent_id, feat_ent_id, now))
                
                # Link Customer Segments
                for aud in target_audience:
                    if not aud:
                        continue
                    aud_key = ("CustomerSegment", aud.strip())
                    if aud_key not in other_entities:
                        aud_id = str(uuid.uuid4())
                        cursor.execute("""
                            INSERT OR IGNORE INTO knowledge_entities 
                            (id, entity_type, canonical_name, source_table, source_record_id, canonicalization_status, attributes_json, created_at, updated_at)
                            VALUES (?, 'CustomerSegment', ?, 'companies', NULL, 'confirmed', '{}', ?, ?)
                        """, (aud_id, aud.strip(), now, now))
                        cursor.execute("SELECT id FROM knowledge_entities WHERE entity_type = 'CustomerSegment' AND canonical_name = ?", (aud.strip(),))
                        row = cursor.fetchone()
                        other_entities[aud_key] = row[0] if row else aud_id
                        
                    aud_ent_id = other_entities[aud_key]
                    prod_lookup["customer_segments"].append(aud.strip())
                    
                    rel_id = str(uuid.uuid4())
                    cursor.execute("""
                        INSERT OR IGNORE INTO knowledge_relationships 
                        (id, source_entity_id, relationship_type, target_entity_id, evidence_confidence, relationship_strength, inference_type, status, evidence_text, valid_from)
                        VALUES (?, ?, 'TARGETS', ?, 1.0, 1.0, 'extracted', 'active', 'Extracted from target audience.', ?)
                    """, (rel_id, prod_ent_id, aud_ent_id, now))

    # 7. Generate Derived Relationships (SIMILAR_TO, COMPETES_WITH)
    print("\n[INGEST] Calculating similar and competing products (Derived relationships)...")
    product_keys = list(product_attributes_lookup.keys())
    similar_links_added = 0
    competitor_links_added = 0

    # Helper: keyword overlap score between two lists of strings
    def _keyword_overlap(list_a, list_b):
        stop = {"and", "or", "the", "for", "to", "a", "an", "in", "of", "with",
                "that", "are", "is", "who", "use", "using", "want", "need"}
        def words(lst):
            w = set()
            for s in lst:
                w.update(t.lower() for t in s.split() if len(t) > 3 and t.lower() not in stop)
            return w
        wa, wb = words(list_a), words(list_b)
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / max(len(wa), len(wb))

    for i in range(len(product_keys)):
        p1_id   = product_keys[i]
        p1_data = product_attributes_lookup[p1_id]
        p1_cats = set(p1_data["categories"])
        p1_aud  = p1_data["customer_segments"]   # list, for keyword overlap
        p1_feats = set(p1_data["features"])

        for j in range(i + 1, len(product_keys)):
            p2_id   = product_keys[j]
            p2_data = product_attributes_lookup[p2_id]
            p2_cats = set(p2_data["categories"])
            p2_aud  = p2_data["customer_segments"]
            p2_feats = set(p2_data["features"])

            # ── Category and Feature Jaccard ──────────────────────────────
            shared_cats  = p1_cats & p2_cats
            sim_c = len(shared_cats) / max(len(p1_cats), len(p2_cats)) if (p1_cats and p2_cats) else 0.0

            shared_feats = p1_feats & p2_feats
            sim_f = len(shared_feats) / max(len(p1_feats), len(p2_feats)) if (p1_feats and p2_feats) else 0.0

            # ── SIMILAR_TO: weighted category (60%) + feature (40%) ───────
            combined_sim = 0.6 * sim_c + 0.4 * sim_f
            if combined_sim < 0.15 and sim_f < 0.30:
                if not shared_cats:
                    continue

            if combined_sim >= 0.12 or sim_f >= 0.25:
                evidence_parts = []
                if shared_cats:
                    evidence_parts.append("Categories: " + ", ".join(list(shared_cats)[:3]))
                if shared_feats:
                    evidence_parts.append("Features: " + ", ".join(list(shared_feats)[:2]))
                rel_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT OR IGNORE INTO knowledge_relationships
                    (id, source_entity_id, relationship_type, target_entity_id, evidence_confidence, relationship_strength, inference_type, status, evidence_text, valid_from)
                    VALUES (?, ?, 'SIMILAR_TO', ?, 0.85, ?, 'derived', 'active', ?, ?)
                """, (rel_id, p1_id, p2_id, round(combined_sim, 3),
                      " | ".join(evidence_parts) or "Shared profile", now))
                similar_links_added += 1

            # ── COMPETES_WITH ─────────────────────────────────────────────
            # Rule A: High category Jaccard (≥ 0.40) alone is sufficient
            if sim_c >= 0.40:
                rel_id_comp = str(uuid.uuid4())
                cursor.execute("""
                    INSERT OR IGNORE INTO knowledge_relationships
                    (id, source_entity_id, relationship_type, target_entity_id, evidence_confidence, relationship_strength, inference_type, status, evidence_text, valid_from)
                    VALUES (?, ?, 'COMPETES_WITH', ?, 0.85, ?, 'derived', 'active', ?, ?)
                """, (rel_id_comp, p1_id, p2_id, round(sim_c, 3),
                      "High category overlap: " + ", ".join(list(shared_cats)[:3]), now))
                competitor_links_added += 1
            # Rule B: Any category overlap + audience keyword overlap (≥ 20%)
            elif shared_cats:
                aud_kw_score = _keyword_overlap(p1_aud, p2_aud)
                if aud_kw_score >= 0.20:
                    cs = min(0.5 * sim_c + 0.5 * aud_kw_score, 1.0)
                    rel_id_comp = str(uuid.uuid4())
                    cursor.execute("""
                        INSERT OR IGNORE INTO knowledge_relationships
                        (id, source_entity_id, relationship_type, target_entity_id, evidence_confidence, relationship_strength, inference_type, status, evidence_text, valid_from)
                        VALUES (?, ?, 'COMPETES_WITH', ?, 0.90, ?, 'derived', 'active', ?, ?)
                    """, (rel_id_comp, p1_id, p2_id, round(cs, 3),
                          "Shared categories: " + ", ".join(list(shared_cats)[:2]) +
                          " | Audience keyword overlap: " + str(round(aud_kw_score, 2)), now))
                    competitor_links_added += 1

    conn.commit()
    conn.close()
    
    print("\n=== INGESTION SUMMARY ===")
    print(f"Total Registered Companies:      {len(company_mapping)}")
    print(f"Total Registered Products:       {len(product_mapping)}")
    print(f"Total Shared Entities (Cat/Feat): {len(other_entities)}")
    print(f"Derived SIMILAR_TO links:        {similar_links_added}")
    print(f"Derived COMPETES_WITH links:     {competitor_links_added}")
    print("Database `knowledge_graph.db` build complete.")

if __name__ == "__main__":
    main()
