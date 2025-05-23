# ReSonnet

> Claude.ai's built-in search only supports matching on conversation titles with no ability to search message content.

ReSonnet is a lightweight vector search that creates a searchable memory of all your past interactions on Claude.ai. It provides a human-friendly search interface as well as function calling capabilities, effectively transforming `claude-3-7-sonnet` into an infinite-turn companion by enabling contextual retrieval of previous conversations.

## Get Sarted
1. **Export**: Export `conversations.json` from Claude.ai
    ![export](https://github.com/user-attachments/assets/e0ccd6d1-b3ff-4f3e-95db-a0f6be3b11ae)
...

## Why is Searching Chat History Challenging?

- Many chats with Claude are throw-away conversations that create noise in retrieval
- Claude generates a LOT of text with extensive explanations and pleasantries
- Conversations often contain debugging/iterative attempts that aren't meaningful long-term
- Users often write unclear or context-dependent prompts leading to undesired assistant messages
- Assistant messages tend to be 50x longer than user messages
- Follow-up questions rely heavily on conversation context that are difficult to capture well in embeddings
- To create an effective infinite-memory companion, retrieval must mirror human memory patterns (Recency, conceptual linking, ...)

#### Agent-driven queries

A major challenge for infintie-context agents is being able to retrieve context with more abstract relevance to the user message that semantic and keyword matching might miss.

#### Building and effective Vector DB

The Vector DB is optimised for smaller scale and is run on-device. There is 1 DB per user. A typical claude.ai user might have 2500 conversations / 15000 messages.

For the user (human) search interface, we expect the following query types:

- Specific entity / knowledge retrieval (keyword)
    - `"react useEffect"` \
Fragmented as no one searches in phrases.
    - `"CRISPR ethics"` \
Often Domain Specific. The user may be target several sessions with this entity for some task.
    - `"Retrieval Ranking Phase Vespa LLM use"` \
Multiple assosciative fragments (often appended to main query)
- Half-remembered information / paraphrasing(semantic)
    - `"vacation spots europe"` \
People often remember they discussed something and want to recover the chat session.
    - `"TikTok Hook for fitness client marketing"` \
Using synonyms or related terminology / Describing the concept rather than using exact terms.
- Queries tageting multiple sessions the searcher uses for knowledge assembly
    - `"Material for Harvard Applciaiton"`
- Referencing Time, unique element of Chat UI, or other inferred context
    - `"Long explainer of Apple types in the UK"` \
Refers to session where a detailed artefact report was generated.
    - `"Debug for AWS deployment error"` \
This user wants to find the code block this was finally resolved in. The code block may be an incremental change with the entireity of the fix only known if combined with the original file. The correct assistant answer may not have any positive feedback: has the user closed the session because they've gotten what they needed, or have they given up?

**SPLADE** wins if: The priority is fast, keyword-driven searches and if storage and speed are critical. It’s also better for short, term-specific queries common in chats.
**Dense embeddings** win if: We want to search by meaning or intent or need to connect related ideas across a conversation. They’re ideal for usage by LLMs for this 'infinite context' effect. Also preffered for smaller data sets

To cover all of these, a hyrbid approach is preffered. This requires an inverted index, a dense vector DB, and a fusion function for determining the most relevant results.

Embedding and Ranking models have to be small as they are again loaded and used on-device.

#### Asisstant Side Prompting / Expose memory retrieval as a callable function

## Areas to explore:
- Matryoshka Embeddings (Although embedding models are already small, and retrieval speed is less important)
- Semantic Retrieval Approaches for Chat History
- FastEmbed
- [Semantic Retrieval at Walmart](https://arxiv.org/abs/2412.04637)
- Splade for keyword matching
- Why Anthropic hasn't released their own search
- Conversation level embedding that is queried first 
- Knowledge Graph of Conversations and Tiered Retrieval Strategy
- [CogGRAG](https://arxiv.org/abs/2503.06567)
- [Local Vector Database with RxDB and transformers.js in JavaScript](https://rxdb.info/articles/javascript-vector-database.html)

