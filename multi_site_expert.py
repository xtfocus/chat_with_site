"""
multi_site_expert.py

Expert agent to answer questions using multiple documentation sources
"""

from __future__ import annotations as _annotations

import os
from dataclasses import dataclass
from typing import List

import logfire
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from supabase import Client

load_dotenv()

llm = os.getenv("LLM_MODEL", "gpt-4o-mini")
model = OpenAIModel(llm)

logfire.configure(send_to_logfire="if-token-present")


@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI
    document_sources: List


system_prompt = """
You are an expert software engineer. You have access to all the documentation to,
including examples, an API reference, and other resources to help you build LLM-related applications.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

Decide based on the provided document sources which ones are relevant to current context.
When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
"""

multi_site_expert = Agent(
    model, system_prompt=system_prompt, deps_type=PydanticAIDeps, retries=2
)


async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error


@multi_site_expert.tool
async def retrieve_relevant_documentation_all(
    ctx: RunContext[PydanticAIDeps], user_query: str
) -> str:
    """
    Retrieve relevant documentation chunks based on the query across all active sources.

    This tool loops over every source listed in `ctx.deps.document_sources`,
    calls the single-source retrieval tool, and aggregates the results.

    Args:
        ctx: The context including the Supabase client and OpenAI client.
        user_query: The user's question or query.

    Returns:
        A combined string with the documentation results from all sources.
    """
    results = []
    for source in ctx.deps.document_sources:
        logger.debug(f"Searching from {source} with query {user_query}")
        res = await retrieve_relevant_documentation(ctx, user_query, source)
        results.append(f"Source: {source}\n{res}")
    return "\n\n==========\n\n".join(results)


# @multi_site_expert.tool
async def retrieve_relevant_documentation(
    ctx: RunContext[PydanticAIDeps], user_query: str, source: str
) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.

    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        source: source corresponding to the document base.

    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)

        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            "match_site_pages",
            {
                "query_embedding": query_embedding,
                "match_count": 5,  # Top number of documents
                "filter": {"source": source},
            },
        ).execute()

        if not result.data:
            return "No relevant documentation found."

        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)

        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"


@multi_site_expert.tool
async def list_documentation_pages(
    ctx: RunContext[PydanticAIDeps], source: str
) -> List[str]:
    """
    Retrieve a list of all available pages.
    Args:
        ctx: The context including the Supabase client and OpenAI client
        source: source corresponding to the document base.

    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is `source`
        result = (
            ctx.deps.supabase.from_("site_pages")
            .select("url")
            .eq("metadata->>source", source)
            .execute()
        )

        if not result.data:
            return []

        # Extract unique URLs
        urls = sorted(set(doc["url"] for doc in result.data))
        return urls

    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []


@multi_site_expert.tool
async def get_page_content(
    ctx: RunContext[PydanticAIDeps], url: str, source: str
) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.

    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        source: source corresponding to the document base.

    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = (
            ctx.deps.supabase.from_("site_pages")
            .select("title, content, chunk_number")
            .eq("url", url)
            .eq("metadata->>source", source)
            .order("chunk_number")
            .execute()
        )

        if not result.data:
            return f"No content found for URL: {url}"

        # Format the page with its title and all chunks
        page_title = result.data[0]["title"].split(" - ")[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]

        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk["content"])

        # Join everything together
        return "\n\n".join(formatted_content)

    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
