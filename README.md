# RAG Assistant (LangChain)

A small Retrieval-Augmented Generation (RAG) app that answers questions from my own text files, built with LangChain and the Gemini API.

This is the second version of my RAG project. The first one was built from scratch (own chunking, embeddings and cosine search). This one rebuilds it with LangChain so I can compare the two and understand what the library does for you.

## How it works

1. **Load**: reads every `.txt` file in the `docs/` folder.
2. **Chunk**: splits each file into ~500-character chunks with a 50-character overlap using `RecursiveCharacterTextSplitter`, which splits on paragraphs and sentences before words. Each chunk keeps its source file name as metadata.
3. **Embed and store**: turns the chunks into embeddings with Gemini (`gemini-embedding-001`) and stores them in LangChain's `InMemoryVectorStore`.
4. **Retrieve**: embeds the question and finds the 3 most similar chunks.
5. **Generate**: sends the question and the retrieved chunks to a Gemini chat model, with instructions to answer only from the context and say "I don't know" otherwise.
6. **Sources**: prints which files the answer came from.

If the model is busy (HTTP 503 on the free tier), the call is retried up to 3 times.

## Setup

1. Clone the repo and create a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_key_here
   ```
4. Put your `.txt` files in the `docs/` folder.

## Run

```
python main.py
```

Change the `question` variable in `main.py` to ask something else.

## Tech

- Python
- LangChain (`langchain`, `langchain-text-splitters`, `langchain-google-genai`)
- Google Gemini API (embeddings and chat model)
- NumPy (used by the vector store for similarity search)

## Limitations

- The vector store is in memory, so documents are re-embedded every time the script runs.
- Only `.txt` files are supported.
- The question is hard-coded in the script.
- No evaluation yet of whether retrieval returns the right chunks.

## Next steps

- Add a small test set of questions to measure retrieval quality
- Save the vector store so it doesn't re-embed on every run
- Add a tool to turn it into a simple agent with LangGraph