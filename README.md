(Most code is copied from https://github.com/coleam00/ottomator-agents/blob/main/crawl4AI-agent/README.md with changes to allow for multi-sources)

# Chat with sites

![screenshot](images/chat_screenshot.jpg)

Dockertized and augmented version of [ottomator](https://github.com/coleam00/ottomator-agents/blob/main/crawl4AI-agent/README.md)

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


### Streamlit Web Interface

For an interactive web interface to query the documentation:

```bash
docker exec -it crawl4ai-python_env-1 sh -c "streamlit run streamlit_ui.py"
```

**Notes:** I met RLS-related errror when trying to insert rows, so I simply went to `supabase.com/dashboard/project/yourprojid/auth/policies` to disable RLS. After the ingesting is complete, I turn it back on. It's lame but I cannot be bothered soon.

## Functionalities:

- UI for ingesting from sitemap.xml urls
- UI for chat
- Select/Deselect multi sources to chat with

Status color:
- green: ready
- blue: ingesting

## Disclaimer

I do not support unethical crawling. This repo is for education only.


## Next plan:
Scope of search:
- Allow selecting github repos to chat with as well

Search and generate config:
- Allow setting common search and text generation config (models, max_tokens, temperature, top_k, etc)

Agentic RAG:
- More complexities: decomposition, delegation, looping
- Tools

