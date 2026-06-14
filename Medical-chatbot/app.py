from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import os
from dotenv import load_dotenv
from functools import lru_cache

app = Flask(__name__)

load_dotenv() 

OPEN_API_KEY = os.getenv('OPEN_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

if not OPEN_API_KEY:
    print("❌ ERROR: OPEN_API_KEY is empty!")
    exit()

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",  # Don't use threshold while debugging
    search_kwargs={"k": 3}    
)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview", 
    google_api_key=OPEN_API_KEY, 
    temperature=0.3,
    max_retries=6,
    timeout=60
)
system_prompt = (
    "You are a professional medical historian and assistant. "
    "1. Use your internal knowledge to briefly explain the historical background of the disease mentioned. "
    "2. Use the PROVIDED CONTEXT to list specific medicines and treatments. "
    "3. Structure your response to be exactly 8 to 9 lines. "
    "If the context doesn't mention the specific disease, use the context to identify what is being discussed and then provide the historical overview and medicinal details."
    "\n\n"
    "CONTEXT: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('index.html', user_input=None, bot_response=None)

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return "Please type a message."
    
    try:

        response = rag_chain.invoke({"input": msg})

        answer = response.get("answer")
        
        if not answer:
            answer = "I'm sorry, I couldn't find any information on that."

        print(f"User: {msg} | Bot: {answer}")

        return str(answer)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return "Error connecting to AI."

@lru_cache(maxsize=100)
def cached_rag_call(query):
    return rag_chain.invoke({"input": query})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)