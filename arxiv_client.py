"""
arxiv_client.py
----------------
Thin wrapper around the public arXiv API (http://export.arxiv.org/api/query).
No API key required. Returns a list of paper dicts: title, authors, summary, url, published.

NOTE: arxiv.org is not reachable from the sandbox this code was written in
(network allowlist restriction), so this has NOT been live-tested here.
It follows arXiv's documented Atom API exactly -- run it in your own
environment (which will have normal internet access) and it should work.
If anything breaks, the most likely culprit is XML namespace handling below.
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import ssl
import certifi

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"


def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """
    Search arXiv for a query string. Returns a list of dicts:
    {title, authors, summary, url, published}
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"

    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, timeout=15, context=context) as response:
            raw_xml = response.read()
    except Exception as e:
        raise RuntimeError(f"arXiv API request failed: {e}")

    return _parse_atom_feed(raw_xml)


def _parse_atom_feed(raw_xml: bytes) -> list[dict]:
    root = ET.fromstring(raw_xml)
    papers = []

    for entry in root.findall(f"{ATOM_NS}entry"):
        title_el = entry.find(f"{ATOM_NS}title")
        summary_el = entry.find(f"{ATOM_NS}summary")
        id_el = entry.find(f"{ATOM_NS}id")
        published_el = entry.find(f"{ATOM_NS}published")

        authors = [
            a.find(f"{ATOM_NS}name").text
            for a in entry.findall(f"{ATOM_NS}author")
            if a.find(f"{ATOM_NS}name") is not None
        ]

        papers.append({
            "title": (title_el.text or "").strip().replace("\n", " ") if title_el is not None else "Untitled",
            "authors": authors,
            "summary": (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else "",
            "url": id_el.text.strip() if id_el is not None else "",
            "published": published_el.text.strip() if published_el is not None else "",
        })

    return papers


if __name__ == "__main__":
    # Quick manual smoke test -- run this file directly in an environment
    # with real internet access to confirm arXiv connectivity works.
    results = search_arxiv("retrieval augmented generation", max_results=3)
    for r in results:
        print("-", r["title"])
        print("  ", r["url"])
