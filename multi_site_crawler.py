"""
multi_site_crawler.py

Provision a SitemapIndexer object that can perform
        Crawling > Chunking > Indexing
        using one sitemap_url

"""

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests
from crawl4ai import (AsyncWebCrawler, BrowserConfig, CacheMode,
                      CrawlerRunConfig)
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel
from supabase import Client, create_client

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY")
)


@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


class TextSplitter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        pass


class BasicTextSplitter(TextSplitter):
    def __init__(self, chunk_size: int, overlapping: int = 0):
        self.chunk_size = chunk_size
        self.overlapping = overlapping

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks, respecting code blocks and paragraphs."""
        chunks = []
        start = 0
        text_length = len(text)

        chunk_size = self.chunk_size

        # TODO: overlapping mechanism
        # Implementing overlapping witll affect the  `get_page_content`` tool of the expert
        # overlapping = self.overlapping

        while start < text_length:
            # Calculate end position
            end = start + chunk_size

            # If we're at the end of the text, just take what's left
            if end >= text_length:
                chunks.append(text[start:].strip())
                break

            # Try to find a code block boundary first (```)
            chunk = text[start:end]
            code_block = chunk.rfind("```")
            if code_block != -1 and code_block > chunk_size * 0.3:
                end = start + code_block

            # If no code block, try to break at a paragraph
            elif "\n\n" in chunk:
                # Find the last paragraph break
                last_break = chunk.rfind("\n\n")
                if (
                    last_break > chunk_size * 0.3
                ):  # Only break if we're past 30% of chunk_size
                    end = start + last_break

            # If no paragraph break, try to break at a sentence
            elif ". " in chunk:
                # Find the last sentence break
                last_period = chunk.rfind(". ")
                if (
                    last_period > chunk_size * 0.3
                ):  # Only break if we're past 30% of chunk_size
                    end = start + last_period + 1

            # Extract chunk and clean it up
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position for next chunk
            start = max(start + 1, end)

        return chunks


class SupabaseURLIndexer:
    """
    Perform indexing on a single url's text
    """

    class SummaryFormat(BaseModel):
        title: str
        summary: str

    def __init__(self, openai_client: AsyncOpenAI, splitter: TextSplitter) -> None:
        self.splitter = splitter
        self.openai_client = openai_client

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector from OpenAI."""
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small", input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return [0] * 1536  # Return zero vector on error

    async def get_title_and_summary(self, chunk: str, url: str) -> "SummaryFormat":
        """Extract title and summary using GPT-4."""
        system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
        Return a JSON object with 'title' and 'summary' keys.
        For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
        For the summary: Create a concise summary of the main points in this chunk.
        Keep both title and summary concise but informative."""

        try:

            response = await self.openai_client.beta.chat.completions.parse(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}...",
                    },  # Send first 1000 chars for context
                ],
                response_format=SupabaseURLIndexer.SummaryFormat,
            )

            parsed = response.choices[0].message.parsed
            logger.debug(parsed)

            if not isinstance(parsed, SupabaseURLIndexer.SummaryFormat):
                raise ValueError("LLM failed at producing structured format")

            return parsed

        except Exception as e:
            logger.error(f"Error getting title and summary: {e}")
            return SupabaseURLIndexer.SummaryFormat(
                title="Error processing title",
                summary="Error processing summary",
            )

    async def process_chunk(
        self, chunk: str, chunk_number: int, url: str, source: str = "general"
    ) -> ProcessedChunk:
        """
        Process a single chunk of text.

        Get chunk title
        Get chunk summary
        Create Chunk metatata

        Return ProcessedChunk object
        """
        # Get title and summary. Using LLM
        extracted: SupabaseURLIndexer.SummaryFormat = await self.get_title_and_summary(
            chunk, url
        )

        # Get embedding
        embedding = await self.get_embedding(chunk)

        # Create metadata
        metadata = {
            "source": source,
            "chunk_size": len(chunk),
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "url_path": urlparse(url).path,
        }

        return ProcessedChunk(
            url=url,
            chunk_number=chunk_number,
            title=extracted.title,
            summary=extracted.summary,
            content=chunk,  # Store the original chunk content
            metadata=metadata,
            embedding=embedding,
        )

    @staticmethod
    async def insert_chunk(chunk: ProcessedChunk):
        """Insert a processed chunk into Supabase."""
        try:
            data = {
                "url": chunk.url,
                "chunk_number": chunk.chunk_number,
                "title": chunk.title,
                "summary": chunk.summary,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "embedding": chunk.embedding,
            }

            result = supabase.table("site_pages").insert(data).execute()
            logger.info(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
            return result
        except Exception as e:
            logger.error(f"Error inserting chunk: {e}")
            # raise
            return None

    async def process_and_store_document(
        self, url: str, markdown: str, source: str
    ) -> None:
        """Process a document and store its chunks in parallel."""
        # Split into chunks
        chunks = self.splitter.chunk_text(markdown)

        # Process chunks in parallel
        tasks = [
            self.process_chunk(chunk, i, url, source=source)
            for i, chunk in enumerate(chunks)
        ]
        processed_chunks = await asyncio.gather(*tasks)

        # Store chunks in parallel
        insert_tasks = [self.insert_chunk(chunk) for chunk in processed_chunks]
        await asyncio.gather(*insert_tasks)


# Refactor some code above to create a class that responsible for crawling from a sitemap_url
class SitemapIndexer:
    """
    Indexing executor for a single sitemap_url
    """

    def __init__(self, indexer: SupabaseURLIndexer):
        self.indexer = indexer

    async def run(self, sitemap_url: str):
        """
        Crawling > Chunking > Indexing
        """
        urls = self.get_docs_urls(sitemap_url)
        if not urls:
            logger.warning("No URLs found to crawl")
            return

        # To test
        # urls = urls[:3]

        logger.info(f"Will crawl {len(urls)} url")
        await self.crawl_parallel(urls, source=sitemap_url)
        return

    @staticmethod
    def get_docs_urls(sitemap_url: str) -> List[str]:
        """Get URLs from sitemap."""
        try:
            response = requests.get(sitemap_url)
            response.raise_for_status()

            # Parse the XML
            root = ElementTree.fromstring(response.content)

            # Extract all URLs from the sitemap
            namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            urls = [loc.text for loc in root.findall(".//ns:loc", namespace)]

            if not urls:
                raise ValueError(f"No urls can be extracted from {sitemap_url}")

            return urls

        except Exception as e:
            logger.error(f"Error fetching sitemap: {e}")
            return []

    async def crawl_parallel(
        self, urls: List[str], max_concurrent: int = 5, source: str = "general"
    ):
        """Crawl multiple URLs in parallel with a concurrency limit."""
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
        )
        crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

        # Create the crawler instance
        crawler = AsyncWebCrawler(config=browser_config)
        await crawler.start()

        try:
            # Create a semaphore to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_url(url: str):
                async with semaphore:
                    result = await crawler.arun(
                        url=url, config=crawl_config, session_id="session1"
                    )
                    if result.success:
                        logger.info(f"Successfully crawled: {url}")
                        await self.indexer.process_and_store_document(
                            url, result.markdown_v2.raw_markdown, source
                        )
                    else:
                        logger.error(f"Failed: {url} - Error: {result.error_message}")

            # Process all URLs in parallel with limited concurrency
            await asyncio.gather(*[process_url(url) for url in urls])
        finally:
            await crawler.close()


async def get_distinct_sources() -> List[str]:
    """
    Fetch distinct 'source' values from the 'metadata' column in the 'site_pages' table.

    Returns:
        List[str]: A list of distinct source values.
    """
    try:
        # Query to select distinct 'source' values from the 'metadata' column
        data = supabase.table("site_pages").select("metadata->source").execute().data

        # Extract the 'source' values from the response
        sources = [item.get("source") for item in data]

        # Remove duplicates by converting the list to a set and back to a list
        distinct_sources = list(set(sources))

        return distinct_sources
    except Exception as e:
        logger.error(f"Error fetching distinct sources: {e}. Table is empty?")
        return []


sitemap_indexer = SitemapIndexer(
    indexer=SupabaseURLIndexer(
        openai_client=openai_client,
        splitter=BasicTextSplitter(chunk_size=5000, overlapping=0),
    )
)

if __name__ == "__main__":
    logger.info("Working example")

    async def main():
        await sitemap_indexer.run(
            "https://ai.pydantic.dev/sitemap.xml",
        )

    asyncio.run(main())
