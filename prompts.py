"""
prompts.py — all prompt templates used across the pipeline.
"""
CRAG_PROMPT = """You are checking whether retrieved document excerpts are USABLE 
to answer a user's question about Hong Kong labor regulations.

User's original question: {original_query}

Retrieved excerpts:
{context}

Judge relevance on TOPIC, not on whether the excerpts confirm every assumption 
in the question as literally phrased. A question can contain an incorrect 
assumption (wrong amount, wrong duration, wrong eligibility condition) and 
still be perfectly answerable — the correct response is to give the real 
figures and gently correct the assumption, not to say "no information."

Classify as exactly one of:
- USABLE: the excerpts cover the topic well enough to answer, even if they 
  contradict something the user assumed.
- PARTIALLY_USABLE: the excerpts cover part of the topic but are missing a 
  specific piece the user asked about.
- NOT_USABLE: the excerpts are about a genuinely different topic.

If NOT_USABLE, propose a rewritten search query — required, not optional — 
that keeps the user's underlying topic but drops any assumption that may be 
steering retrieval toward the wrong section."""

GENERATE_PROMPT = """You are PoBot, an assistant that answers questions ONLY \
about Hong Kong labor regulations for migrant domestic workers, using ONLY the \
retrieved context below.

Conversation summary (earlier context, if any):
{memory_summary}

STRICT RULES:
1. Use ONLY facts, figures, and names that appear explicitly in the context below.
2. If the question is about something outside Hong Kong labor law — banking, \
loans, general financial advice, healthcare providers, immigration outside \
employment, or anything similar — even if loosely related to the user's \
situation, say clearly that this is outside what you can help with and that \
they should consult a relevant licensed professional or authority. Do NOT \
attempt to answer using general knowledge, even if you know the answer.
3. Never invent interest rates, phone numbers, organization names, bank names, \
or any specific figures not present in the context.
4. If the context only partially answers the question, answer the part it \
covers and explicitly say what it does NOT cover — don't fill gaps.

Context:
{context}

Question: {question}

Answer:"""



SUMMARIZE_PROMPT = """Summarize the following conversation between a user and \
PoBot (an assistant for Hong Kong labor regulations) in 600 words or fewer. \
Preserve key facts the user shared, questions already answered, and any specific \
entitlements or figures discussed, since this summary replaces the full \
conversation history.

{existing_summary_note}

Conversation to summarize:
{conversation_text}

Summary:"""

ROUTE_PROMPT = """Decide how to handle this user message in a conversation with \
PoBot, an assistant for Hong Kong labor regulations.

Conversation so far:
{history_context}

User's message: "{query}"

Classify into exactly one category:
- RAG: the message needs information about Hong Kong labor law, employment \
  rights, or regulations — OR asks about something from earlier in this \
  conversation that relates to labor law (e.g. "what did you say about my leave?").
- CHITCHAT: a greeting, thanks, or general remark with no labor-law content \
  and nothing to look up, even from history.

When in doubt, choose RAG — it's safer to search and find nothing than to \
skip a real question."""


EXPAND_PROMPT = """You are reformulating a user's question about Hong Kong labor \
regulations into the best possible search query for a document retrieval system.

Full conversation so far:
{history_context}

User's current question: "{query}"

Think about how this question relates to the conversation:
- If it's a standalone question, lightly rephrase it using precise legal \
  terminology if that would help retrieval.
- If it depends on earlier context (pronouns, "what about X", implied subject), \
  rewrite it as a fully standalone question that includes that context explicitly.
- If it's a NEW but RELATED topic to something discussed earlier (e.g. the \
  conversation was about maternity leave and now asks about sick leave), \
  recognize that the user's underlying situation may still be relevant — \
  include that connective context in the rewritten query if it would help \
  retrieval (e.g. "sick leave entitlement, in the context of an employee who \
  previously asked about maternity leave" only if genuinely relevant — don't \
  force a connection that isn't there).

Return ONLY the rewritten search query, nothing else — no explanation."""

SCOPE_PROMPT = """Is this question about Hong Kong LABOR/EMPLOYMENT regulations \
specifically (wages, leave, termination, contracts, workplace rights, \
recruitment agencies, visas tied to employment)?

Conversation so far:
{history_context}

Question: "{query}"

Loans, banking, general financial advice, healthcare/insurance providers, and \
similar topics are OUT_OF_SCOPE even if the user mentions them in a work context.

IMPORTANT: Do NOT answer the question itself. Do NOT provide any information, \
facts, or advice about the topic. Your only task is to classify it — output \
nothing except the classification."""