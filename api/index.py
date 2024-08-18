from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
#importing
from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, WebBaseLoader, YoutubeLoader, DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import numpy as np
import tiktoken
from groq import Groq
import os
from pydantic import BaseModel
from dotenv import load_dotenv
import json
from typing import List  # Make sure to import List from typing

#loading keys and env variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify the domains you want to allow
    allow_credentials=True,
    allow_methods=["*"],  # or specify the methods you want to allow
    allow_headers=["*"],  # or specify the headers you want to allow
)

#initializing groq client to send api request
client = Groq(api_key=groq_api_key)

# instructions to final model (llama 3.1 8b)
primer = f"""You are a personal assistant. Answer any questions I have about the Youtube Video provided.
Translate in specific language if user asks you to
"""

#loading video from youtube
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=e-gwvmhyU7A", add_video_info=True)
def split_transcript(transcript, max_chunk_size=10000):
    chunks = []
    current_chunk = ""
    
    for line in transcript.split("\n"):
        if len(current_chunk) + len(line) > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += "\n" + line
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# Usage
transcript = loader.load()  # Assume this loads the transcript
data = split_transcript(transcript)


# making chunks of data got from youtube video
tokenizer = tiktoken.get_encoding('p50k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

#defining text_splitter to make chunks
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
)

# splitting data from youtube video using text splitter and tokenizer
texts = text_splitter.split_documents(data)

#defining hugging face embedding variable
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#initializing vector database "Chroma db"
vector_store = Chroma(
    collection_name="data_collection",
    embedding_function=hf_embeddings,
)

#storing data into documents variable to add to chroma db (vector database)

documents= [
    Document(
        page_content=f"Source: {t.metadata['source']}, Title: {t.metadata['title']} \n\nContent: {t.page_content}",
                   metadata=t.metadata
                   )
    for t in texts]

#adding to database
vectorstore_from_texts = vector_store.add_documents(documents=documents)

# function to create vector embeddings of the user query 

def get_embedding(text, model="sentence-transformers/all-MiniLM-L6-v2"):
    response= HuggingFaceEmbeddings(model_name=model)
    embedded_response = hf_embeddings.embed_query(text)
    return embedded_response


#making api
# Define the QueryRequest model to accept a list of messages
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    messages: List[Message]



@app.post('/api/ask/')


#function for getting response from the llm
async def get_response(request:QueryRequest):
  print('api started running')
  try:
    all_messages= request.messages
 #calling the function for user query vector embeddings
    raw_query_embedding=get_embedding(all_messages[-1].content)
    
    #after converting query to vector, performing similarity search by vector 
    results = vector_store.similarity_search_by_vector(
        embedding=raw_query_embedding, k=1
    )
    
    #providing the context to llm by performing similarity vector search
    contexts = [doc.page_content for doc in results]
    
    #maintain user chat history
    conversation_history= "\n\n".join(f"{msg.role}: {msg.content}" for msg in all_messages)
    msg_string=json.dumps(conversation_history)
    #combining context with user query to allow model to perform RAG
    augmented_query = (
            "<CONTEXT>\n" +
            "\n\n-------\n\n".join(contexts) +
            "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" +
            conversation_history
        )
    
    # makng call to llm to get the answer
    response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": augmented_query},
            ],
            max_tokens=1000,
            temperature=1.2)
    
    #returning the response of llm
    return {'assistantMessage':response.choices[0].message.content}
  except Exception as e:
      raise HTTPException(status_code=500,detail= str(e))




