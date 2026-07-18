"""
Website Content HTML-to-Markdown Cleaner
Parses raw HTML fragments and returns clean, structured Markdown, removing ads,
navigation headers, footers, cookie popups, and script blocks to optimize AI processing tokens.
"""
import re
from bs4 import BeautifulSoup, Comment

def clean_html_to_markdown(html_content: str) -> str:
    """
    Extracts visible text and formats standard HTML tags into Markdown.
    
    Args:
        html_content (str): Raw HTML code.
        
    Returns:
        str: AI-ready Markdown content.
    """
    if not html_content or not html_content.strip():
        return ""

    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Strip comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # 2. Decompose completely useless structural tags
    useless_tags = [
        "script", "style", "svg", "noscript", "iframe", "embed", "object",
        "header", "footer", "nav", "aside", "form", "dialog", "button",
        "select", "option", "textarea", "input"
    ]
    for tag in soup(useless_tags):
        tag.decompose()

    # 3. Decompose cookie banners, GDPR notifications, ads, and popups
    unwanted_patterns = [
        r"(?:cookie|banner|consent|gdpr|popup|modal|ad-|promo|advert|overlay|newsletter|subscribe)"
    ]
    combined_pattern = re.compile("|".join(unwanted_patterns), re.IGNORECASE)

    for element in soup.find_all(True):
        # Examine element id and class list attributes
        attrs = [element.get("id"), *element.get("class", [])]
        attrs = [str(a).strip().lower() for a in attrs if a]
        
        # If class/id matches target patterns, remove the block
        if any(combined_pattern.search(attr) for attr in attrs):
            element.decompose()

    # 4. Convert structural HTML to Markdown tags
    
    # Headers h1 to h6
    for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        level = int(h.name[1])
        h.replace_with(f"\n\n{'#' * level} {h.get_text().strip()}\n\n")

    # Bold/Strong
    for b in soup.find_all(["strong", "b"]):
        text = b.get_text().strip()
        if text:
            b.replace_with(f" **{text}** ")

    # Lists
    for li in soup.find_all("li"):
        text = li.get_text().strip()
        if text:
            li.replace_with(f"\n- {text}")

    # Anchor Links (convert to markdown link style)
    for a in soup.find_all("a", href=True):
        text = a.get_text().strip()
        href = a.get("href").strip()
        # Keep link only if it looks like an actual useful web link and has text
        if text and href and href.startswith(("http", "/")):
            a.replace_with(f" [{text}]({href}) ")

    # Paragraphs (add line breaks)
    for p in soup.find_all("p"):
        text = p.get_text().strip()
        if text:
            p.replace_with(f"\n\n{text}\n\n")

    # 5. Extract compiled text and clean spaces
    raw_text = soup.get_text()
    
    # Clean multiple spaces and blank lines
    lines = [line.strip() for line in raw_text.splitlines()]
    
    # Collapse consecutive blank lines
    cleaned_lines = []
    prev_blank = False
    for line in lines:
        if line == "":
            if not prev_blank:
                cleaned_lines.append("")
                prev_blank = True
        else:
            cleaned_lines.append(line)
            prev_blank = False

    final_markdown = "\n".join(cleaned_lines).strip()
    
    # Clean up double line breaks and duplicate whitespaces
    final_markdown = re.sub(r'[ \t]+', ' ', final_markdown)
    final_markdown = re.sub(r'\n{3,}', '\n\n', final_markdown)
    
    return final_markdown
