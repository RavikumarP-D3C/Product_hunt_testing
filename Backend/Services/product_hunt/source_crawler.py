"""
Source Crawler Interface
Defines the base class interface for crawlers harvesting data from various sources (e.g. Website, GitHub, LinkedIn, etc.)
"""
from typing import Dict, Any

class BaseCrawler:
    """
    Base class for all crawl services.
    Enforces a standard crawl signature for plug-and-play platform compatibility.
    """
    
    async def crawl(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the crawl action on a target resource/URL.
        
        Args:
            url (str): Target URL or resource identifier.
            
        Returns:
            Dict[str, Any]: Standardized crawl result dict containing content, metadata, and status.
        """
        raise NotImplementedError("Subclasses must implement the crawl method.")
