import time
from typing import List, Dict, Optional

from Bio import Entrez
from bs4 import BeautifulSoup

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Required by NCBI
Entrez.email = settings.NCBI_EMAIL
Entrez.tool = "health_assistant_graph_rag"


def search_pmc_articles(query: str, max_results: int) -> List[str]:
    """
    Search PubMed Central (PMC) for open-access full-text articles.
    """
    full_query = f"{query} AND open access[filter]"
    logger.info("Searching PMC", extra={"query": query, "max_results": max_results})

    try:
        handle = Entrez.esearch(
            db="pmc",
            term=full_query,
            retmax=max_results,
        )
        record = Entrez.read(handle)
        handle.close()

        pmc_ids = record.get("IdList", [])
        logger.info("PMC search complete", extra={"results": len(pmc_ids)})
        return pmc_ids

    except Exception:
        logger.exception("PMC search failed")
        return []


def _extract_full_text(soup: BeautifulSoup) -> str:
    """
    Extract readable full text from PMC XML <body>.
    """
    body = soup.find("body")
    if not body:
        return ""

    sections = []
    for sec in body.find_all("sec"):
        title_tag = sec.find("title")
        section_title = title_tag.get_text(strip=True) if title_tag else "Section"
        paragraphs = [p.get_text(strip=True) for p in sec.find_all("p")]

        if paragraphs:
            sections.append(f"## {section_title}\n" + "\n".join(paragraphs))

    return "\n\n".join(sections)


def fetch_pmc_details(pmc_id: str) -> Optional[Dict]:
    """
    Fetch and parse a single PMC full-text article.
    """
    try:
        handle = Entrez.efetch(
            db="pmc",
            id=pmc_id,
            rettype="full",
            retmode="xml",
        )
        xml_data = handle.read()
        handle.close()

        soup = BeautifulSoup(xml_data, "lxml-xml")

        article = {
            "pmid": pmc_id,
            "title": "No Title",
            "journal": "Unknown Journal",
            "year": 0,
            "abstract": "",
            "study_type": "Full Text Article",
        }

        title_tag = soup.find("article-title")
        if title_tag:
            article["title"] = title_tag.get_text(strip=True)

        journal_tag = soup.find("journal-title")
        if journal_tag:
            article["journal"] = journal_tag.get_text(strip=True)

        pub_date = soup.find("pub-date", {"pub-type": "epub"}) or soup.find("pub-date")
        if pub_date and pub_date.find("year"):
            article["year"] = int(pub_date.find("year").get_text())

        full_text = _extract_full_text(soup)
        article["abstract"] = full_text

        return article

    except Exception:
        logger.exception("Failed to fetch PMC article", extra={"pmc_id": pmc_id})
        return None


def fetch_all_pmc_articles(query: str, max_results: int = 5) -> List[Dict]:
    """
    End-to-end PMC fetch pipeline.
    """
    pmc_ids = search_pmc_articles(query, max_results)
    articles: List[Dict] = []

    for pmc_id in pmc_ids:
        article = fetch_pmc_details(pmc_id)
        if article and len(article.get("abstract", "")) > settings.MIN_TEXT_LENGTH:
            articles.append(article)
            logger.info("PMC article processed", extra={"pmc_id": pmc_id})

        time.sleep(settings.NCBI_REQUEST_DELAY)

    logger.info("PMC ingestion complete", extra={"articles": len(articles)})
    return articles