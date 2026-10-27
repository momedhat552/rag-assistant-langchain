from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

def chunker(file):
    with open(file, encoding="utf-8") as f:
        text = f.read() 
        splitter = RecursiveCharacterTextSplitter(chunk_size =500, chunk_overlap=50)
        chunks = splitter.create_documents([text], metadatas=[{"source": str(file)}])
        return chunks

all_chunks = []

for path in Path("docs").glob("*.txt"):
    chunks = chunker(path)
    all_chunks.extend(chunks)

store = InMemoryVectorStore(embeddings)
store.add_documents(all_chunks)

question = "what is a docker?"
results = store.similarity_search(question, k=3)

context = "\n\n".join(doc.page_content for doc in results)

prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}"""

response = llm.invoke(prompt)
print(response.text)
print("sources: ", {doc.metadata["source"] for doc in results})

