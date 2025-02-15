import asyncio
import os
from dataclasses import dataclass
from typing import Literal, TypedDict

import streamlit as st
from loguru import logger
from openai import AsyncOpenAI
from pydantic_ai.messages import (ModelMessage, ModelMessagesTypeAdapter,
                                  ModelRequest, ModelResponse, RetryPromptPart,
                                  SystemPromptPart, TextPart, ToolCallPart,
                                  ToolReturnPart, UserPromptPart)
from supabase import Client

from multi_site_crawler import (BasicTextSplitter, SitemapIndexer,
                                SupabaseURLIndexer, get_distinct_sources,
                                sitemap_indexer)
from multi_site_expert import PydanticAIDeps, multi_site_expert

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))


class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal["user", "model"]
    timestamp: str
    content: str


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "urls" not in st.session_state:
        st.session_state.urls = []
    if "url_status" not in st.session_state:
        st.session_state.url_status = {}
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_sources" not in st.session_state:
        st.session_state.document_sources = []


def display_message_part(part, container):
    """Display a single message part in the Streamlit UI within the specified container."""
    if part.part_kind == "system-prompt":
        with container.chat_message("system"):
            container.markdown(f"**System**: {part.content}")
    elif part.part_kind == "user-prompt":
        with container.chat_message("user"):
            container.markdown(part.content)
    elif part.part_kind == "text":
        with container.chat_message("assistant"):
            container.markdown(part.content)


async def crawl_url(url, index):
    """Crawl a URL and update its status."""
    try:
        await sitemap_indexer.run(url)
        st.session_state.urls[index]["status"] = "completed"
    except Exception as e:
        st.error(f"Error crawling {url}: {e}")
        st.session_state.urls[index]["status"] = "error"
    st.rerun()


def render_url_section():
    """Render the URL management section."""
    st.markdown("### Enter sitemap urls!")

    sources = asyncio.run(get_distinct_sources())

    # Display existing sources
    for i, source in enumerate(sources):
        row_col1, row_col2 = st.columns([4, 0.5])
        with row_col1:
            selected = st.checkbox(source, key=f"source_{i}", value=True)

            if selected and source not in st.session_state.document_sources:
                st.session_state.document_sources.append(source)
            elif not selected and source in st.session_state.document_sources:
                st.session_state.document_sources.remove(source)

        with row_col2:
            st.markdown(
                "<span style='color: green; font-size: 24px;'>●</span>",
                unsafe_allow_html=True,
            )

    # Add new URL button
    if st.button("＋ Add"):
        st.session_state.urls.append({"url": "", "status": "pending"})

    # Display URL entries
    for i, url_data in enumerate(st.session_state.urls):
        render_url_entry(i, url_data)


def render_url_entry(index, url_data):
    """Render a single URL entry with its status indicator."""
    row_col1, row_col2 = st.columns([4, 0.5])
    with row_col1:
        new_url = st.text_input(
            label=f"URL {index + 1}",
            value=url_data["url"],
            placeholder="Enter new sitemap URL",
            label_visibility="collapsed",
            key=f"new_url_{index}",
        )
        st.session_state.urls[index]["url"] = new_url

    with row_col2:
        render_url_status(url_data, index, new_url)


def render_url_status(url_data, index, url):
    """Render the status indicator for a URL."""
    if url_data["status"] == "pending":
        if st.button("➤", key=f"new_url_arrow_{index}"):
            st.session_state.urls[index]["status"] = "processing"
            st.rerun()
    elif url_data["status"] == "processing":
        st.markdown(
            "<span style='color: blue; font-size: 24px;'>●</span>",
            unsafe_allow_html=True,
        )
        asyncio.run(crawl_url(url, index))
    elif url_data["status"] == "completed":
        st.markdown(
            "<span style='color: green; font-size: 24px;'>●</span>",
            unsafe_allow_html=True,
        )


async def run_agent_with_streaming(user_input: str, message_container):
    """Run the agent with streaming response."""
    # Gather selected document sources
    document_sources = st.session_state.get("document_sources", [])

    logger.debug(f"document_sources = {document_sources}")

    deps = PydanticAIDeps(
        supabase=supabase,
        openai_client=openai_client,
        document_sources=document_sources,  # Pass selected sources here
    )

    async with multi_site_expert.run_stream(
        user_input, deps=deps, message_history=st.session_state.messages[:-1]
    ) as result:
        partial_text = ""
        message_placeholder = message_container.empty()

        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        filtered_messages = [
            msg
            for msg in result.new_messages()
            if not (
                hasattr(msg, "parts")
                and any(part.part_kind == "user-prompt" for part in msg.parts)
            )
        ]
        st.session_state.messages.extend(filtered_messages)
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )


def render_chat_interface():
    """Render the chat interface with proper layout."""
    st.markdown("### Chat with sites")

    # Create a container for the entire chat interface
    chat_area = st.container()

    # Create a container for messages that will scroll
    message_container = chat_area.container()

    # Add some spacing to prevent the last message from being hidden
    message_container.markdown(
        "<div style='height: 100px'></div>", unsafe_allow_html=True
    )

    # Display all previous messages in the scrolling container
    for msg in st.session_state.messages:
        if isinstance(msg, (ModelRequest, ModelResponse)):
            for part in msg.parts:
                display_message_part(part, message_container)

    # Create a container for the input at the bottom
    input_container = chat_area.container()

    # Add the chat input in the bottom container
    user_input = input_container.chat_input(
        "What questions do you have about Pydantic AI?"
    )

    if user_input:
        # Add user message to history
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )

        # Display user message
        with message_container.chat_message("user"):
            message_container.markdown(user_input)

        # Display assistant response
        with message_container.chat_message("assistant"):
            asyncio.run(run_agent_with_streaming(user_input, message_container))

        # Rerun to update the UI
        st.rerun()


def main():
    """Main application entry point."""
    st.set_page_config(layout="wide")
    initialize_session_state()

    # Create the main layout
    url_col, chat_col = st.columns([1, 2])

    # Render URL section in left column
    with url_col:
        render_url_section()

    # Render chat interface in right column
    with chat_col:
        render_chat_interface()


if __name__ == "__main__":
    main()
