# `ReSonnet`

A lightweight vector search tool for your chat history on claude.ai! Built for Humans and Agents alike.
- Creates a searchable memory of all past interactions
- Allows `claude-3-7-sonnet` to retrieve all messages relevant to the current conversation, transforming Claude into a true **infinite-turn companion**.

> ReSonnet additionally offers a human-friendly search interface. \
> Claude.ai current limits search to keyword matching on conversation title.


## Why is Chat History Search Challenging?

- Many chats with Claude are throw-away conversations that create noise in retrieval
- Conversations often contain debugging/iterative attempts that aren't meaningful long-term
- Claude generates a LOT of text with extensive explanations, boilerplate text and pleasantries
- To create an effective infinite-memory companion, retrieval must mirror human memory patterns (Recency, conceptual linking, ...)
- Users often write unclear or context-dependent prompts leading to undesired assistant messages
- Assistant messages tend to be 50x longer than user messages
- Follow-up questions rely heavily on conversation context that may not be captured well in embeddings

#### Agent-driven queries

A major challenge for infintie-context agents is being able to retrieve context with more abstract relevance to the user message that semantic and keyword matching might miss. [Human-like Episodic Memory for Infinite Context LLMs](https://arxiv.org/abs/2407.09450) introduces EM-LLM, a novel approach that attempts to replicate human memory to solve this problem. EM-LLM organises and stores sequences of tokens into coherent episodic events based on suprise boundaries, and combines similarity-based and temporally contiguous retrieval for efficient and human-like retrieval.

#### Building and effective Vector DB

The Vector DB is optimised for smaller scale and is run on-device. There is 1 DB per user. A typical claude.ai user might have 2500 conversations / 15000 messages.
For the user search interface, we expect queries about specific entities (e.g. 'Email to Lucas Vannevar') and queries intending to find a specific chat session, thus a hyrbid sparse (keyword) and semantic (dense) approach is preffered. 

This will work with the epsiodic chunking approach preffered for Agent generated queries and will require an inverted index, a dense vector DB, and a fusion function for determining the most relevant results. 

During indexing we can either approach the chat histroy as a series of episdoes as outlines above or structued arround conversations. 
Embedding and Ranking models have to be small as they are again loaded and used on-device.

## Areas to explore:
- Matryoshka Embeddings (Although embedding models are already small, and retrieval speed is less important)
- Semantic Retrieval Approaches for Chat History
- FastEmbed
- [Semantic Retrieval at Walmart](https://arxiv.org/abs/2412.04637)
- Splade for keyword matching
- Why Anthropic hasn't released their own search
- Conversation drive search - conversation level embedding that is queried first 


## Installing / Usage

1. **Export**: Export `conversations.json` from Claude.ai
    ![export](https://github.com/user-attachments/assets/e0ccd6d1-b3ff-4f3e-95db-a0f6be3b11ae)
2. **Index**: Process and embed all conversations
3. **Search**: During conversations, Claude can query this database to recall relevant context
