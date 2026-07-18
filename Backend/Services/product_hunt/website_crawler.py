"""
Website Crawler Implementation
Implements a modular crawling architecture including LinkExtractor, MetadataExtractor,
ContentFetcher, SnapshotBuilder, and CompanyResolver.
"""
import os
import re
import time
import json
import logging
import hashlib
import urllib.robotparser
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse, urljoin
import requests
# pyrefly: ignore [missing-import]
from bs4 import BeautifulSoup

from Utils.llm_client import get_llm_client
from Services.product_hunt.source_crawler import BaseCrawler

logger = logging.getLogger(__name__)

class CompanyResolver:
    """
    Resolves the parent company name from product name, description, website, and crawled homepage metadata.
    Uses metadata/footer parsing as primary and LLM as fallback.
    """
    
    @staticmethod
    def parse_footer_copyright(html_soup: BeautifulSoup) -> Optional[str]:
        """
        Attempts to find copyright holder info in footers.
        """
        # Search for text containing © or Copyright
        copyright_pattern = re.compile(r'(?:copyright|©)\s*(?:\d{4}-)?\d{4}\s+([^•\|\n©\.\,]{2,100})', re.IGNORECASE)
        for element in html_soup.find_all(string=True):
            if "©" in element or "copyright" in element.lower():
                parent = element.parent
                # Ensure it looks like a footer/nav or bottom element
                if parent and parent.name in ["footer", "div", "p", "span", "a"]:
                    text = element.strip()
                    match = copyright_pattern.search(text)
                    if match:
                        company = match.group(1).strip()
                        # Clean up common suffixes
                        company = re.sub(r'\s*(?:inc|llc|ltd|co|corporation|corp|all rights reserved|\.)[\s\,]*$', '', company, flags=re.IGNORECASE)
                        if len(company) > 1:
                            return company
        return None

    async def resolve_company_name(
        self, 
        product_name: str, 
        website: str, 
        description: str, 
        homepage_html: str
    ) -> str:
        """
        Resolves the legal/parent company name.
        """
        soup = BeautifulSoup(homepage_html, "html.parser")
        
        # 1. Primary Flow: Try Footer Copyright
        footer_company = self.parse_footer_copyright(soup)
        if footer_company and len(footer_company) > 1:
            logger.info(f"CompanyResolver: Found company name '{footer_company}' in homepage footer.")
            return footer_company

        # 2. Try OG Tags / Title suffix
        og_site_name = ""
        og_site_meta = soup.find("meta", property="og:site_name")
        if og_site_meta and og_site_meta.get("content"):
            og_site_name = og_site_meta.get("content").strip()
            
        title_text = ""
        if soup.title and soup.title.string:
            title_text = soup.title.string.strip()
            
        # If OG site name exists and doesn't equal the product name, it might be the company
        if og_site_name and og_site_name.lower() != product_name.lower():
            logger.info(f"CompanyResolver: Found company name '{og_site_name}' in og:site_name.")
            return og_site_name

        # 3. Fallback Flow: LLM classification
        token = os.getenv("AZURE_OPENAI_API_KEY")
        if not token:
            logger.warning("CompanyResolver: No LLM credentials. Defaulting to product_name.")
            return product_name

        model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
        prompt = f"""
        Given the following product details, resolve the official parent / legal company name behind the product.
        E.g. Product: Cursor IDE -> Company: Anysphere. Product: ChatGPT -> Company: OpenAI.
        
        Product Name: {product_name}
        Website: {website}
        Description: {description}
        Homepage Title: {title_text}
        
        Return a JSON object containing the resolved company name in the field "company_name".
        Example: {{"company_name": "Anysphere"}}
        """
        
        try:
            client = get_llm_client()
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a company name resolution assistant. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(response.choices[0].message.content)
            company_name = data.get("company_name", product_name).strip()
            logger.info(f"CompanyResolver: LLM resolved company name to '{company_name}'")
            return company_name
        except Exception as e:
            logger.error(f"CompanyResolver: LLM fallback resolution failed: {e}")
            return product_name


class LinkExtractor:
    """
    Extracts and prioritizes internal links from the homepage.
    Uses Rule Engine preprocessing to filter links and LLM to choose the final top 7.
    """
    
    @staticmethod
    def is_internal(url: str, base_url: str) -> bool:
        parsed_base = urlparse(base_url)
        parsed_url = urlparse(url)
        if not parsed_url.netloc:  # relative link
            return True
        return parsed_url.netloc == parsed_base.netloc

    def extract_internal_links(self, homepage_html: str, base_url: str) -> List[str]:
        soup = BeautifulSoup(homepage_html, "html.parser")
        links = []
        for tag in soup.find_all("a", href=True):
            href = tag.get("href")
            # Resolve relative links
            full_url = urljoin(base_url, href)
            # Remove hash query
            full_url = full_url.split("#")[0].split("?")[0].rstrip("/")
            if self.is_internal(full_url, base_url) and full_url != base_url:
                if full_url.startswith("http"):
                    links.append(full_url)
        return list(set(links))

    def filter_with_rule_engine(self, links: List[str]) -> List[str]:
        """
        Removes obviously useless pages like privacy, cookies, login, status.
        """
        useless_patterns = re.compile(
            r'(?:login|signin|signup|register|privacy|cookies|cookie-settings|terms|tos|status|careers|jobs|support-center|help-center|contact|download)', 
            re.IGNORECASE
        )
        candidates = []
        for l in links:
            if not useless_patterns.search(l):
                candidates.append(l)
        return candidates

    async def classify_links(self, candidates: List[str], base_url: str) -> List[str]:
        """
        Uses Rule Engine + LLM classification to select the top 7 most informative pages.
        """
        # Filter candidate links
        filtered = self.filter_with_rule_engine(candidates)
        
        # If we have very few links, crawl them all without LLM
        if len(filtered) <= 7:
            return filtered

        # Cap candidates to top 20 to avoid large token size
        # Prioritize keywords to put interesting URLs first
        priority_patterns = re.compile(r'(?:pricing|features|solutions|docs|faq|about|integrations|blog)', re.IGNORECASE)
        filtered.sort(key=lambda x: 0 if priority_patterns.search(x) else 1)
        top_candidates = filtered[:20]

        token = os.getenv("AZURE_OPENAI_API_KEY")
        if not token:
            logger.warning("LinkExtractor: No LLM credentials. Falling back to rule-based link selection.")
            return top_candidates[:7]

        model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
        prompt = f"""
        Select up to 7 most relevant subpages of a company website to crawl.
        We want to analyze the company's product, pricing, solutions, documentation, integrations and core offerings.
        Select ONLY from the list below.
        
        Base Website: {base_url}
        Candidate URLs:
        {json.dumps(top_candidates, indent=2)}
        
        Return a JSON object containing the selected URLs in the field "urls" (list of strings).
        Example: {{"urls": ["https://site.com/pricing"]}}
        """
        
        try:
            client = get_llm_client()
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a web crawler link classification assistant. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(response.choices[0].message.content)
            urls = data.get("urls", [])
            return [u for u in urls if u in top_candidates]
        except Exception as e:
            logger.error(f"LinkExtractor: LLM link classification failed: {e}")
            return top_candidates[:7]


class MetadataExtractor:
    """
    Extracts rich website SEO, OpenGraph, JSON-LD, sitemaps, favicons and language.
    """
    @staticmethod
    def extract_metadata(html: str, base_url: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        meta_dict = {
            "title": "",
            "description": "",
            "canonical": "",
            "language": "",
            "favicon": "",
            "robots": "",
            "opengraph": {},
            "twitter": {},
            "json_ld": []
        }

        # Title
        if soup.title and soup.title.string:
            meta_dict["title"] = soup.title.string.strip()

        # HTML Lang
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            meta_dict["language"] = str(html_tag.get("lang"))

        # Canonical URL
        canon_tag = soup.find("link", rel="canonical")
        if canon_tag and canon_tag.get("href"):
            meta_dict["canonical"] = urljoin(base_url, canon_tag.get("href"))

        # Favicon Link
        fav_tag = soup.find("link", rel=lambda x: x and ("icon" in x.lower() or "shortcut" in x.lower()))
        if fav_tag and fav_tag.get("href"):
            meta_dict["favicon"] = urljoin(base_url, fav_tag.get("href"))

        # Meta tags
        for meta in soup.find_all("meta"):
            name = meta.get("name")
            prop = meta.get("property")
            content = meta.get("content", "")

            if name == "description":
                meta_dict["description"] = content
            elif name == "robots":
                meta_dict["robots"] = content
            elif prop and prop.startswith("og:"):
                key = prop[3:]
                meta_dict["opengraph"][key] = content
            elif name and name.startswith("twitter:"):
                key = name[8:]
                meta_dict["twitter"][key] = content

        # JSON-LD schemas
        for script in soup.find_all("script", type="application/ld+json"):
            if script.string:
                try:
                    meta_dict["json_ld"].append(json.loads(script.string.strip()))
                except Exception:
                    pass

        return meta_dict


class ContentFetcher:
    """
    Fetches raw HTML from URLs politely respecting robots.txt and adding rate limits.
    """
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self._robots_parsers: Dict[str, urllib.robotparser.RobotFileParser] = {}

    def _get_robots_parser(self, base_url: str) -> urllib.robotparser.RobotFileParser:
        parsed = urlparse(base_url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        if root not in self._robots_parsers:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(urljoin(root, "/robots.txt"))
            try:
                # Do a quick read of robots.txt
                response = self.session.get(urljoin(root, "/robots.txt"), timeout=5)
                if response.status_code == 200:
                    rp.parse(response.text.splitlines())
                else:
                    rp.parse([])
            except Exception:
                rp.parse([])
            self._robots_parsers[root] = rp
        return self._robots_parsers[root]

    def is_allowed(self, url: str) -> bool:
        """Checks if URL crawling is allowed under robots.txt directive."""
        rp = self._get_robots_parser(url)
        return rp.can_fetch("*", url)

    def fetch_page_html(self, url: str) -> Optional[str]:
        """
        Fetch HTML page politely.
        """
        if not self.is_allowed(url):
            logger.warning(f"ContentFetcher: Crawl disallowed by robots.txt: {url}")
            return None

        # Polite crawler sleep
        time.sleep(0.5)
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"ContentFetcher: Error fetching {url}: {e}")
            return None


class SnapshotBuilder:
    """
    Builds the combined snapshot hash and compiles the raw MongoDB structure.
    """
    @staticmethod
    def calculate_combined_hash(raw_content: Dict[str, str]) -> str:
        """
        Combines the HTML contents of homepage and subpages, and calculates a SHA-256 hash.
        """
        # Sort keys to ensure deterministic hashing
        hash_input = ""
        for key in sorted(raw_content.keys()):
            hash_input += raw_content[key]
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


class WebsiteCrawler(BaseCrawler):
    """
    Orchestrated Website Crawler.
    Uses sub-classes for Company Resolution, Link Extraction, Metadata Extraction, Fetching, and Snapshot Building.
    """
    
    def __init__(self):
        self.company_resolver = CompanyResolver()
        self.link_extractor = LinkExtractor()
        self.metadata_extractor = MetadataExtractor()
        self.content_fetcher = ContentFetcher()
        self.snapshot_builder = SnapshotBuilder()

    async def crawl(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Crawl a target website URL dynamically resolving company name and extracting subpages.
        
        Args:
            url (str): Root/homepage URL of target company.
            kwargs:
                product_name (str): Product Hunt name
                description (str): Product Hunt description
                
        Returns:
            Dict[str, Any]: Standardized crawl output matching v4 specification.
        """
        product_name = kwargs.get("product_name", "")
        description = kwargs.get("description", "")
        
        logger.info(f"WebsiteCrawler: Starting crawl for {url}")
        
        # 1. Fetch homepage
        homepage_html = self.content_fetcher.fetch_page_html(url)
        if not homepage_html:
            return {"status": "error", "error": f"Failed to fetch homepage for {url}"}
            
        # 2. Resolve official company name
        company_name = await self.company_resolver.resolve_company_name(
            product_name, url, description, homepage_html
        )
        
        # 3. Extract metadata from homepage
        metadata = self.metadata_extractor.extract_metadata(homepage_html, url)
        metadata["company_name"] = company_name
        
        # 4. Extract and classify subpages
        internal_links = self.link_extractor.extract_internal_links(homepage_html, url)
        selected_subpages = await self.link_extractor.classify_links(internal_links, url)
        
        # 5. Fetch subpages politely
        raw_content = {
            "homepage": homepage_html
        }
        
        # Map URL suffix (e.g. /pricing -> pricing) as key for dictionary
        for page_url in selected_subpages:
            parsed = urlparse(page_url)
            # Create a safe name for key
            key = parsed.path.strip("/").replace("/", "_")
            if not key:
                key = "index"
            
            logger.info(f"WebsiteCrawler: Crawling subpage {page_url} (Key: {key})")
            html = self.content_fetcher.fetch_page_html(page_url)
            if html:
                raw_content[key] = html
                
        # 6. Build combined SHA-256 hash
        combined_hash = self.snapshot_builder.calculate_combined_hash(raw_content)
        
        return {
            "status": "success",
            "company_name": company_name,
            "website": url,
            "metadata": metadata,
            "raw_content": raw_content,
            "combined_hash": combined_hash
        }
