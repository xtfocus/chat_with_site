Codes copied from https://github.com/coleam00/ottomator-agents/blob/main/crawl4AI-agent/README.md with minor changes

# Deployment

## Requirements
- Create a project at supabase, acquire url and api_key
- OpenAI API KEY
- Put secrets in `.env` (see `.env.example`)

## Usage

Start the container:
```bash
docker compose up --build -d
````

### Database Setup

Execute the SQL commands in `site_pages.sql` to:
1. Create the necessary tables
2. Enable vector similarity search
3. Set up Row Level Security policies

In Supabase, do this by going to the "SQL Editor" tab and pasting in the SQL into the editor there. Then click "Run".

### Crawl Documentation

To crawl and store documentation in the vector database:

```bash
docker exec -it crawl4ai-python_env-1 sh -c "python crawl_pydantic_ai_docs.py"
```

This will:
1. Fetch URLs from the documentation sitemap
2. Crawl each page and split into chunks
3. Generate embeddings and store in Supabase (Using OpenAI)

**Notes:** At this step I met RLS-related errror, so I simply went to `supabase.com/dashboard/project/prjid/auth/policies` to disable RLS. After the ingesting is complete, I turn it back on. It's lame but I cannot be bothered soon.

### Streamlit Web Interface

For an interactive web interface to query the documentation:

```bash
docker exec -d crawl4ai-python_env-1 sh -c "streamlit run streamlit_ui.py
```

# Next plan:

Scope of search:
- Allow selecting multiple sites to chat with at once
- Allow selecting github repos to chat with as well

Search and generate config:
- Allow setting common search and text generation config (models, max_tokens, temperature, top_k, etc)

Agentic RAG:
- More complexities: decomposition, delegation, looping
- Tools


