# PoBot — Hong Kong Labor Regulations Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Hong Kong
labor and employment regulations for migrant domestic workers, grounded strictly in
official government and legal source documents.

## Approach & Findings (Summary)

PoBot combines hierarchical, structure-aware PDF chunking with a hybrid BM25 +
dense-embedding retrieval pipeline, a Cohere reranker, and a Corrective-RAG (CRAG)
grading loop that catches both irrelevant retrieval and false-premise questions before
generation. Answers are grounded with mandatory, deterministically-verified citations
back to source headings and filenames — the model can forget to cite a fact, but it
cannot fabricate a source, since citations are rebuilt from chunk metadata rather than
trusted from the model's own output. The system is orchestrated as a LangGraph state
machine (route → expand → retrieve → CRAG → generate, with a chitchat side-path and a
fallback path) rather than a single linear chain, which made it straightforward to add
and remove control-flow nodes (e.g. a dedicated scope-check node) as a deliberate
cost/reliability tradeoff rather than a prompt-engineering afterthought. The main known
weakness is documented below rather than hidden: one internal call site
(`_update_summary`) is intentionally left without exception handling to serve as the
project's documented failure case.

## Architecture

```
                         ┌─────────┐
              START ───▶ │  route  │  (RAG vs CHITCHAT)
                         └────┬────┘
                   ┌──────────┴──────────┐
                   ▼                     ▼
             ┌──────────┐          ┌───────────┐
             │  expand  │          │ chitchat  │──▶ END
             └────┬─────┘          └───────────┘
                   ▼
             ┌──────────┐
             │ retrieve │  (hybrid BM25 + FAISS → Cohere rerank)
             └────┬─────┘
                   ▼
             ┌──────────┐   not relevant, retries left
             │   crag   │ ─────────────────┐
             └────┬─────┘                  │
        relevant  │        retries exhausted│
                   ▼                        ▼
             ┌──────────┐            ┌──────────┐
             │ generate │──▶ END     │ fallback │──▶ END
             └──────────┘            └──────────┘
```

**Nodes:**

| Node | Responsibility |
|---|---|
| `route` | Classifies the message as `RAG` (needs lookup) or `CHITCHAT` (greeting, small talk). Defaults to `RAG` on classifier failure — safer to search and find nothing than to skip a real question. |
| `chitchat` | Handles greetings/small talk. Still has full conversation memory, so context like "I'm Karim, from Bangladesh, working in HK" is captured here and available to later RAG turns. |
| `expand` | Rewrites the query into a standalone, retrieval-optimized query using conversation history (resolves pronouns, follow-ups, related-topic continuity). |
| `retrieve` | Hybrid retrieval: BM25 (keyword) + dense embeddings (`bge-small-en-v1.5`) fused via min-max normalized scoring, then reranked with Cohere `rerank-english-v3.0`. |
| `crag` | Grades retrieved chunks as `USABLE` / `PARTIALLY_USABLE` / `NOT_USABLE` against the *topic*, not literal question phrasing — so a question with a wrong assumption (wrong amount, wrong eligibility condition) still gets graded usable and corrected, rather than rejected. On `NOT_USABLE`, proposes a rewritten query and retries (max 2 retries). |
| `generate` | Produces the final answer, grounded only in retrieved context. Requires `[CITE:n]` tags on every factual sentence; the final "Sources:" block is rebuilt deterministically from chunk metadata, not trusted from model output. |
| `fallback` | Returned when CRAG retries are exhausted without a usable result — an honest "I don't have reliable information" rather than a guess. |

### On the scope-check node

An earlier version of this pipeline included a dedicated `scope_node` — a pydantic-gated
LLM call that hard-refused out-of-scope topics (banking, healthcare, general finance)
*before* retrieval, rather than relying on the instruction embedded in `GENERATE_PROMPT`
(rule 2). It was removed as a deliberate tradeoff: `crag_node` already filters
off-topic content via its relevance grading, and `GENERATE_PROMPT` rule 2 already
refuses out-of-scope topics at generation time, so the extra LLM call was mostly
redundant latency/cost on every single turn rather than a meaningful reliability gain.
If this system were deployed at higher volume, or if hard topic-boundary guarantees
became a compliance requirement, reintroducing that node (see `SCOPE_PROMPT` in git
history) would be the first thing to add back — it trades a small latency cost for a
control-flow guarantee that doesn't depend on the model reliably following an embedded
instruction.

## Project Structure

```
.
├── pdfs/                    # raw source PDFs (input to cleaning.py)
├── cleaned_texts/           # per-source cleaned .txt with [SOURCE_FILE]/[AUTHORITY] headers
├── cleaning.py              # PDF extraction, per-source cleaning, hierarchical chunking
├── indexing.py              # builds BM25 + FAISS indexes from chunks.jsonl
├── retrieval.py             # hybrid retrieval, Cohere rerank, query expansion
├── prompts.py                # all prompt templates
├── pipeline.py               # LangGraph state machine, memory, CLI entry point
├── chunks.jsonl              # output of cleaning.py (chunk corpus)
├── bm25_index.pkl            # output of indexing.py
├── faiss_index.bin           # output of indexing.py
├── chunk_lookup.pkl          # output of indexing.py
└── README.md
```

## Data Sources

Source documents live in `pdfs/` and are tagged by authority in `cleaning.py`'s
`AUTHORITY_MAP`:

| Document | Authority |
|---|---|
| Employment Ordinance guide (`EO_guide`) | statute |
| Code of Practice (`CoP_Eng`) | gov_guidance |
| Foreign Domestic Helper guide (`FDHguideEnglish`) | gov_guidance |
| Standard Employment Contract | gov_guidance |
| Hong Kong Judiciary – Labour Tribunal guide | gov_guidance |
| `ID(E)969` (immigration/visa form guidance) | gov_guidance |

Add new sources by dropping a PDF into `pdfs/` and, if it needs custom text cleanup,
adding an entry to `SOURCE_CLEANING_RULES` and `AUTHORITY_MAP` in `cleaning.py`.

## Setup

### Requirements

```
pip install pymupdf pdfplumber rank_bm25 sentence-transformers faiss-cpu \
            cohere langchain-groq langgraph pydantic python-dotenv numpy
```

### Environment variables (`.env`)

```
GROQ_API_KEY=your_groq_key
COHERE_API_KEY=your_cohere_key
```

## Running the Pipeline

Run once, in order, to build the corpus and indexes:

```bash
python cleaning.py     # pdfs/ → cleaned_texts/ → chunks.jsonl
python indexing.py     # chunks.jsonl → bm25_index.pkl, faiss_index.bin, chunk_lookup.pkl
```

Then start the chatbot:

```bash
python pipeline.py
```

```
PoBot — Hong Kong Labor Regulations Assistant
Type your question, or 'exit' to quit.

You: What are the rest day entitlements for a domestic helper?
PoBot: ...
```

## Citations

Every generated answer that draws on retrieved context ends with a `Sources:` block,
e.g.:

```
Domestic helpers are entitled to at least one rest day every 7 days [CITE:1]...

Sources:
[CITE:1] 17. Rest Days — EO_guide.pdf
```

`[CITE:n]` tags are attached by `generate_node` to numbered context chunks before
generation, and the final source list is rebuilt after generation from the chunks'
actual metadata (`heading_text`, `source_file`) for whichever tags the model actually
used — this list is never generated freeform by the model, to prevent fabricated
filenames.

## Conversation Memory

`ConversationMemory` retains the last 10 raw messages and rolls older ones into a
running summary (capped at ~600 words) via `SUMMARIZE_PROMPT`. `expand_node`,
`chitchat_node`, `route_node`, and `generate_node` all read from
`memory.get_full_context()`, so context volunteered early in a conversation (e.g. a
user's country of origin or role) is available to later, unrelated-seeming follow-up
questions.

**Known limitation:** memory is currently a single global instance
(`memory = ConversationMemory()` at module scope), not scoped per session/user. Fine
for the single-user CLI this ships with; would need to become a
`dict[session_id, ConversationMemory]` threaded through the graph state before this
could safely serve concurrent users (e.g. behind an API).

## Known Limitations / Failure Cases

- **`ConversationMemory._update_summary` is intentionally unguarded.** If the
  summarization LLM call fails (rate limit, API error), it raises and crashes the
  current turn rather than degrading gracefully. This is left as-is deliberately, as
  the project's documented failure case (see Evaluation below) — every other LLM call
  site in the pipeline (`route`, `chitchat`, `expand`, `crag`, `generate`) has
  exception handling with a safe fallback.
- **Memory is not session-scoped** (see above).
- **No multilingual support** — English only; Tagalog query support was considered
  out of scope for this pass.
- **Out-of-scope refusal relies on `GENERATE_PROMPT` + CRAG topic filtering**, not a
  dedicated pre-retrieval scope classifier (see "On the scope-check node" above).

## Evaluation

See `evaluation/` (or your evaluation doc) for the 3–5 example queries, their actual
chatbot outputs, and the documented failure case writeup required for this
assignment's evaluation criteria.
