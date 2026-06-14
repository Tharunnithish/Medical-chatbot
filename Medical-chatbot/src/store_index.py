system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec 
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file(data='data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

from flask import Flask, render_template, jsonify, request
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from src.prompt import *

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = OpenAI(temperature=0.4, max_tokens=500)

prompt = ChatPromptTemplate.from_messages(
   [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User Input: {msg}")
    
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)