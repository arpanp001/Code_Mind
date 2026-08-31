# 🧠 CodeMind — RAG-Based AI-Powered Codebase Memory System

<div align="center">

![CodeMind Banner](https://img.shields.io/badge/CodeMind-AI%20Codebase%20Intelligence-6366f1?style=for-the-badge&logo=openai&logoColor=white)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://reactjs.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B35?style=flat-square)](https://www.trychroma.com)
[![Gemini](https://img.shields.io/badge/Google-Gemini%202.5%20Flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Upload any GitHub repository or ZIP file and chat with your codebase using AI.**
Find functions, understand architecture, trace request flows, and store architectural decisions — all in one place.

[🚀 Live Demo](#-live-demo) • [✨ Features](#-features) • [🛠️ Tech Stack](#️-tech-stack) • [⚡ Quick Start](#-quick-start) • [📖 How It Works](#-how-it-works)

</div>

---

## 🎯 What is CodeMind?

CodeMind is a full-stack **Retrieval-Augmented Generation (RAG)** application that lets you have intelligent conversations with any codebase. Instead of manually searching through hundreds of files, simply ask questions in plain English and get precise, cited answers.

```
You:  "Where is JWT authentication implemented?"
AI:   "JWT authentication is in src/auth/login.py (lines 42-67).
       The authenticate() function validates credentials and calls
       create_token() which uses the HS256 algorithm..."
       ↳ Source: src/auth/login.py [function] 89% match
       ↳ Source: src/utils/jwt.py  [function] 76% match
```

---

## ✨ Features

### 🔍 Smart Code Retrieval
- **Hybrid Search** — Combines ChromaDB semantic search with BM25+ keyword matching for 5-signal ranking
- **Path-Aware Ranking** — `src/` implementation files ranked above `docs/` automatically
- **Exact Identifier Matching** — Queries for `authenticate()` find the exact function, not just mentions
- **Multi-Query Expansion** — Each question generates 3-4 variations for broader coverage

### 🌳 AST-Based Chunking
- **Complete Logical Units** — Functions and classes are never split mid-body
- **Language Support** — Python (`ast.parse`), JavaScript/TypeScript, Java, Go with brace-counting
- **Rich Metadata** — Every chunk stores function name, class name, start/end lines, decorators
- **Smart Fallback** — Gracefully falls back to regex chunking for unsupported languages

### 🏗️ Architecture Intelligence
- **Dependency Graph** — Analyzes imports, inheritance, and module relationships during indexing
- **Entry Point Detection** — Automatically identifies `main.py`, Flask/FastAPI apps, route handlers
- **Request Flow Tracing** — Ask "How does a request flow?" and get a traced answer
- **Mermaid Diagrams** — Auto-generates architecture diagrams for visual exploration

### 💬 AI-Powered Chat
- **Google Gemini 2.5 Flash** — State-of-the-art LLM for precise, context-aware answers
- **Multi-Turn Conversations** — Session-aware chat with conversation history injection
- **Memory System** — Store architectural decisions, bug fixes, and notes that persist across sessions
- **Source Citations** — Every answer links back to exact file, function, and line numbers

### 📦 Easy Ingestion
- **GitHub URL** — Paste any public repo URL and index it in minutes
- **ZIP Upload** — Upload a local project as a ZIP file
- **Duplicate Detection** — Detects already-indexed repos and offers re-index option
- **Progress Tracking** — Real-time SSE pipeline progress (cloning → parsing → chunking → embedding → storing)

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | React 18 + Vite + Tailwind CSS | Responsive chat UI |
| **Backend** | FastAPI + Uvicorn | REST API + SSE streaming |
| **Vector DB** | ChromaDB 0.6.3 | Semantic similarity search |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) | 384-dim code embeddings |
| **LLM** | Google Gemini 2.5 Flash | Answer generation |
| **Retrieval** | LangChain + BM25+ + RRF | Hybrid search pipeline |
| **Database** | SQLite + aiosqlite | Project registry |
| **Chunking** | Python AST + Brace-counting | Language-aware code splitting |
| **Parsing** | GitPython + pathspec | GitHub cloning + file filtering |

---

## 📸 Screenshots

<div align="center">

### Upload & Index
> Paste a GitHub URL → CodeMind clones, parses, chunks, and indexes the entire codebase

### Chat Interface
> Ask questions in plain English and get AI answers with file citations and relevance scores

### Memory System
> Store architectural decisions that persist across all future conversations

### Architecture View
> Auto-generated dependency graphs and request flow diagrams

</div>

---

## 🚀 Live Demo

🌐 **[https://codemind-arpan.vercel.app](https://code-mind-nu.vercel.app)**

Try it with these example repos:
- `https://github.com/pallets/flask` — Flask web framework (206 files)
- `https://github.com/tiangolo/fastapi` — FastAPI framework
- `https://github.com/django/django` — Django (large repo test)

---

## ⚡ Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Google Gemini API key ([get one free](https://makersuite.google.com/app/apikey))
- Git

---

### 1. Clone the Repository

```bash
git clone https://github.com/arpanp001/Code_Mind
cd codemind
```

---

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create your environment file:

```bash
# Copy the example file
cp .env.example .env
```

Open `backend/.env` and fill in your values:

```ini
# Required
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=models/gemini-2.5-flash

# App settings
APP_ENV=development
FRONTEND_URL=http://localhost:5173

# Storage (leave as defaults)
CHROMA_PERSIST_PATH=./chroma_db
UPLOAD_DIR=./uploads

# RAG settings (leave as defaults)
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K_RESULTS=5
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

Start the backend:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
✅ Database initialized
✅ All systems ready — accepting requests
```

---

### 3. Frontend Setup

Open a new terminal:

```bash
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env
```

Open `frontend/.env`:

```ini
VITE_API_URL=http://localhost:8000
```

Start the frontend:

```bash
npm run dev
```

---

### 4. Open the App

Go to **http://localhost:5173** in your browser.

**Test it:**
1. Paste `https://github.com/pallets/flask` in the GitHub URL box
2. Click **Index Repository**
3. Wait ~2 minutes for indexing
4. Ask: **"What is the entry point of Flask?"**

---

## 📖 How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                    │
│                                                         │
│  GitHub URL / ZIP                                       │
│       ↓                                                 │
│  Clone / Extract  →  Parse Files  →  Detect Language   │
│       ↓                                                 │
│  AST Chunking     →  Generate Embeddings (MiniLM)      │
│       ↓                                                 │
│  ChromaDB Storage →  Dependency Graph Analysis         │
│       ↓                                                 │
│  Architecture Memory (entry points, imports, flows)    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                       │
│                                                         │
│  User Question                                          │
│       ↓                                                 │
│  Query Expansion (3-4 variations)                      │
│       ↓                                                 │
│  Hybrid Search:                                         │
│    • ChromaDB semantic search (55%)                    │
│    • BM25+ keyword match     (25%)                     │
│    • Path quality score      (10%)                     │
│    • Exact identifier match  ( 7%)                     │
│    • Chunk type score        ( 3%)                     │
│       ↓                                                 │
│  RRF Fusion + Re-ranking                               │
│       ↓                                                 │
│  Architecture Context (if structural question)         │
│       ↓                                                 │
│  Gemini 2.5 Flash → Answer with Citations             │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
codemind/
├── backend/
│   ├── app/
│   │   ├── api/routes/
│   │   │   ├── ingest.py       # GitHub/ZIP ingestion endpoints
│   │   │   ├── query.py        # Chat, search, explain endpoints
│   │   │   └── memory.py       # Memory CRUD endpoints
│   │   ├── core/
│   │   │   ├── rag/
│   │   │   │   ├── embedder.py         # Sentence Transformer embeddings
│   │   │   │   ├── vectorstore.py      # ChromaDB client
│   │   │   │   ├── hybrid_search.py    # 5-signal hybrid scorer
│   │   │   │   ├── reranker.py         # Multi-signal re-ranker
│   │   │   │   ├── retriever.py        # Full retrieval pipeline
│   │   │   │   └── context_assembler.py # Token-budget context builder
│   │   │   ├── processing/
│   │   │   │   ├── ast_chunker.py      # AST-based code chunking
│   │   │   │   ├── chunker.py          # Phase 7 fallback chunker
│   │   │   │   ├── parser.py           # File parsing + language detection
│   │   │   │   └── language_detector.py
│   │   │   ├── llm/
│   │   │   │   ├── gemini.py           # Gemini API client
│   │   │   │   ├── rag_generator.py    # Full RAG answer generation
│   │   │   │   ├── prompts.py          # Prompt templates
│   │   │   │   └── conversation_memory.py
│   │   │   ├── analysis/
│   │   │   │   ├── dependency_analyzer.py  # Import/inheritance graph
│   │   │   │   ├── architecture_query.py   # Structural question engine
│   │   │   │   └── architecture_memory.py  # Graph persistence
│   │   │   ├── ingestion/
│   │   │   │   ├── github.py           # GitHub repo cloning
│   │   │   │   ├── zip_handler.py      # ZIP extraction
│   │   │   │   └── github_api.py       # GitHub API preview
│   │   │   └── memory/
│   │   │       └── project_memory.py   # ChromaDB memory store
│   │   ├── models/                     # Pydantic request/response models
│   │   ├── utils/                      # Database, logger helpers
│   │   ├── config.py                   # Pydantic settings
│   │   └── main.py                     # FastAPI app entry point
│   ├── tests/
│   │   ├── test_hybrid_retrieval.py
│   │   ├── test_ast_chunker.py
│   │   └── test_architecture.py
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── HomePage.jsx        # Upload + ingestion UI
│   │   │   ├── ChatPage.jsx        # Main chat interface
│   │   │   └── MemoryPage.jsx      # Memory management
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── Sidebar.jsx     # Project list
│   │   │   │   └── Navbar.jsx
│   │   │   ├── chat/
│   │   │   │   ├── MessageBubble.jsx
│   │   │   │   └── SourceCard.jsx  # File citation cards
│   │   │   └── upload/
│   │   │       ├── GithubInput.jsx
│   │   │       └── ZipUpload.jsx
│   │   ├── hooks/
│   │   │   ├── useIngest.js
│   │   │   └── usePipelineProgress.js
│   │   ├── context/
│   │   │   └── ChatContext.jsx
│   │   └── services/
│   │       ├── api.js
│   │       └── ingestService.js
│   ├── package.json
│   └── .env.example
│
└── README.md
```

---

## 🧪 Running Tests

```bash
cd backend
.venv\Scripts\activate   # Windows
# or: source .venv/bin/activate (Mac/Linux)

# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_hybrid_retrieval.py -v   # Hybrid search tests
pytest tests/test_ast_chunker.py -v        # AST chunking tests
pytest tests/test_architecture.py -v      # Dependency graph tests
```

Expected output:
```
tests/test_hybrid_retrieval.py  ......................  37 passed
tests/test_ast_chunker.py       ......................  54 passed
tests/test_architecture.py      ......................  60 passed
==================== 151 passed in 8.3s ====================
```

---

## 🔧 Environment Variables Reference

### Backend (`backend/.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | — | Google Gemini API key |
| `GEMINI_MODEL` | ✅ Yes | `models/gemini-2.5-flash` | Gemini model name |
| `APP_ENV` | No | `development` | `development` or `production` |
| `FRONTEND_URL` | No | `http://localhost:5173` | CORS allowed origin |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | Sentence Transformer model |
| `CHROMA_PERSIST_PATH` | No | `./chroma_db` | ChromaDB storage path |
| `UPLOAD_DIR` | No | `./uploads` | ZIP upload temp directory |
| `TOP_K_RESULTS` | No | `5` | Number of chunks to retrieve |
| `CHUNK_SIZE` | No | `500` | Target chunk size in chars |
| `CHUNK_OVERLAP` | No | `50` | Overlap between chunks |

### Frontend (`frontend/.env`)

| Variable | Required | Description |
|---|---|---|
| `VITE_API_URL` | ✅ Yes | Backend URL (`http://localhost:8000` for local) |

---

## 🚢 Deployment

### Deploy to Render + Vercel (Free)

**Backend → Render:**
1. Go to [render.com](https://render.com) → New Web Service
2. Connect your GitHub repo
3. Settings:
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add all environment variables from the table above
5. Deploy

**Frontend → Vercel:**
1. Go to [vercel.com](https://vercel.com) → New Project
2. Import your GitHub repo
3. Settings:
   - Root Directory: `frontend`
   - Framework: Vite
4. Add `VITE_API_URL` = your Render backend URL
5. Deploy

---

## 💡 Example Questions to Ask

Once a repo is indexed, try these:

```
Architecture questions:
  "What is the entry point of this application?"
  "How does a request flow through the system?"
  "What does UserService depend on?"
  "Show me the architecture overview"

Code questions:
  "Where is authentication implemented?"
  "How does the database connection work?"
  "What does the create_token() function do?"
  "Find all API endpoints"

Debugging questions:
  "Where are errors handled?"
  "How is logging configured?"
  "Where is the configuration loaded from?"
```

---

## 🗺️ Roadmap

- [ ] Private GitHub repository support (GitHub token auth)
- [ ] Multi-project cross-search
- [ ] VS Code extension
- [ ] Slack integration via Claude Tag
- [ ] Support for TypeScript type inference
- [ ] Export chat conversations as PDF
- [ ] Team collaboration with shared memory

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Google Gemini](https://deepmind.google/technologies/gemini) — LLM powering the AI answers
- [ChromaDB](https://www.trychroma.com) — Vector database for semantic search
- [Sentence Transformers](https://www.sbert.net) — Code embedding model
- [LangChain](https://langchain.com) — RAG pipeline utilities
- [FastAPI](https://fastapi.tiangolo.com) — High-performance Python API framework
- [Anthropic Claude](https://anthropic.com) — AI assistant used during development

---

<div align="center">

**Built with ❤️ by [Arpan Patel](https://github.com/arpanp001)**

⭐ Star this repo if you find it useful!

</div>
