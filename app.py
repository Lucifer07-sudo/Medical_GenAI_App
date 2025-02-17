from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from src.prompt import *

#intialize Flask app
app = Flask(__name__)

#load env variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY='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'



# PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
# OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY


#print("GROQ_API_KEY:", GROQ_API_KEY)

index_name = "medicalbot"

#get embedding model
embeddings = download_huggingface_embeddings()

# Load vectors from existing index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

#create retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

#create LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="Llama3-8b-8192",
    temperature=0,        # Controls randomness (0.0 for deterministic, 1.0 for max creativity)
    max_tokens=500          # Maximum tokens in the response
)


#create prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ('system',system_prompt),
        ('human',"{input}")
    ]
)

#create stuufed chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
#create retriever chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


##Flask part##

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods = ['POST','GET'])
def chat():
    msg = request.form["msg"]                    ## get user querry
    input = msg                
    print(input)                                 ## print user input to coder
    response = rag_chain.invoke({'input':msg})   ## invoke retriever chain to get LLM chain
    print("Response:", response["answer"])       ## print LLM response to coder
    return str(response["answer"])               ## return response to user on chat window


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
