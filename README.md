# ReSonnet

> Claude.ai's built-in search only supports matching on conversation titles with no ability to search message content.

ReSonnet is a lightweight only-when-needed vector search that creates a memory of all your past interactions on Claude.ai. It provides a human-friendly search interface as well as function calling capabilities (MCP), effectively transforming `claude-4-sonnet` into an infinite-turn companion by enabling contextual retrieval of previous conversations.

## Get Sarted
1. **Export**: Export `conversations.json` from Claude.ai
    ![export](https://github.com/user-attachments/assets/e0ccd6d1-b3ff-4f3e-95db-a0f6be3b11ae)

## Why Searching Chat History is Challenging?

- Many chats with Claude are throw-away conversations that create noise in retrieval
- Claude generates a LOT of text with extensive explanations and pleasantries
- Conversations often contain debugging/iterative attempts that aren't meaningful long-term
- Users often write unclear or context-dependent prompts leading to undesired assistant messages
- Assistant messages tend to be 50x longer than user messages
- Noisy pasted context within user messages
- Follow-up questions rely heavily on conversation context that are difficult to capture well in embeddings
- Injecting irrelevant context contaminates responses. Reasoning models are particularly prone to over-incorporating tangential context into the thought process.
- Users create new chat sessions if they are not satisfied with the assistant's response meaning recency is not a good indicator of relevance.
- Minimal signals for relevant context that would be useful for future reference.
- Turn-level is too granular, session-level includes irrelevant content. Topical segmentation is needed for coherent memory.

## Insights from Human Memory

- Emotionally charged events much more likely to be consolidated into long-term memory.
- Increased likelihood to remember information that connects to existing knowledge (elaborative encoding).
- Unexpected/surprising events reach conscious awareness thus are more likely to be remembered.
- Surprise based episode boundaries (see EM-LLM).
- Sleep mechanism that consolidates memories, works as an editorial review filtering what to remember.
- Memory retrieval influenced by environmental context (not availanle to LLMs).
- Group memory often exceeds individual memory through complementary retrieval cues. i.e. Reminiscant night talking on the porch.

## Areas to explore:
- [ ] [Why I no longer recommend RAG for LLM applications](https://pashpashpash.substack.com/p/why-i-no-longer-recommend-rag-for)
- [ ] [Claude Code Team on Ditching RAG](https://open.spotify.com/episode/6ffGB5ter845nffKHzOFqv) - More appliable to coding agents that mirror human engineers (grep, etc).
- [ ] [Semantic Retrieval at Walmart](https://arxiv.org/abs/2412.04637)
- [ ] [CogGRAG](https://arxiv.org/abs/2503.06567)
- [ ] [Local Vector Database with RxDB and transformers.js in JavaScript](https://rxdb.info/articles/javascript-vector-database.html)
- [ ] [Binary Embedding Quantization](https://huggingface.co/blog/embedding-quantization)

A major challenge for infinite-context agents is being able to retrieve context with more abstract relevance to the user message that semantic and keyword matching might miss.
Adaptive Injection of Context to avoid contamination in prompts that dont need it.
Turn-level is too granular, session-level includes irrelevant content. Implement topical segmentation to create coherent memory chunks to make Claude feel like it truly "remembers" your relationshi rather than just searching isolated conversation fragments.

## Building and effective Vector DB

The Vector DB is optimised for smaller scale and is run on-device. There is 1 DB per user. A typical claude.ai user might have 2500 conversations / 15000 messages.

For a user-facing search interface, we expect the following query types:

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

## Memory in ChatGPT

*Reference Saved Memories:* ChatGPT detects and stores useful information for future conversations. This is injected into the context everytime a new non-temporary chat is started. <br/> `Model Set Context:[2025-05-02]. The user dislikes HR style language in generated text.`
*Reference Chat History:* RAG over chat history. Currently, this injects a lot of tokens and can contaminate responses.

[SeCom](https://arxiv.org/abs/2502.05589) Findings: Turn-level is too granular, session-level includes irrelevant content. Implement topical segmentation to create coherent memory chunks to make Claude feel like it truly "remembers" your relationshi rather than just searching isolated conversation fragments.

Vannevar Bush (from "As We May Think," 1945): "The human operates by association. With one item in its grasp, it snaps instantly to the next that is suggested by the association of thoughts, in accordance with some intricate web of trails carried by the cells of the brain."

## Memory in Claude Code

Claude Code recreates how senior engineers work with codebases. They don't read isolated snippets - they keyword search, explore entire files, and reason contextually. Reffered to as agentic discovery by Claude Code lead engineer Boris Cherny.

##  `multilingual-e5-large-instruct` Task Descriptions
```
All purpose:
- "Represent this conversation segment for retrieving contextually relevant discussions about specific topics, decisions, or insights that would be valuable for future reference"
For User Queries:
- "Represent this user query for retrieving relevant past conversations focusing on the core intent and key concepts discussed"
For Assistant Responses:
- "Represent this response for retrieval focusing on actionable insights, solutions, and key information while ignoring conversational pleasantries"
For Agent-Driven Queries:
- "Represent this query for finding conceptually related conversations that may contain relevant context or solutions"
```