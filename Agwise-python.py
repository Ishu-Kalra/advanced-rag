import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import partial
from pydantic import BaseModel, Field
from textstat import textstat

import langchain
from langchain_community.document_loaders import PyPDFLoader
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross Origin Resource Sharing for API access from different domains

# Set API keys and environment variables
api_key = ""
os.environ["OPENAI_API_KEY"] = api_key
os.environ["API_KEY_OPENAI"] = api_key

# Define a directory to store FAISS indices and ensure it exists
FAISS_INDEX_DIR = "./faiss_indexes"
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# Mapping of years to their respective PDF file paths
years = {
    "UNL-Agriculture-2013": "./agwiseapp/2013research-results.pdf",
    "UNL-Agriculture-2014": "./agwiseapp/2014research-results.pdf",
    "UNL-Agriculture-2015": "./agwiseapp/2015research-results.pdf",
    "UNL-Agriculture-2016": "./agwiseapp/2016research-results.pdf",
    "UNL-Agriculture-2017": "./agwiseapp/2017research-results.pdf",
    "UNL-Agriculture-2018": "./agwiseapp/2018research-results.pdf",
    "UNL-Agriculture-2019": "./agwiseapp/2019research-results.pdf",
    "UNL-Agriculture-2020": "./agwiseapp/2020research-results.pdf",
    "UNL-Agriculture-2021": "./agwiseapp/2021research-results.pdf",
    "UNL-Agriculture-2022": "./agwiseapp/2022research-results.pdf",
    "UNL-Agriculture-2023": "./agwiseapp/2023research-results.pdf",
}

# Function to load existing FAISS index or create a new one from documents of a given year
def load_or_create_faiss_index_from_documents(year: str, file_path: str,
                                            embeddings: OpenAIEmbeddings,
                                            text_splitter: RecursiveCharacterTextSplitter,
                                            create_anyway: bool = False) -> FAISS:
    # Define the path to save/load the index for the given year
    index_path = os.path.join(FAISS_INDEX_DIR, f"{year.lower()}_documents")
    
    # If index exists and recreate flag is false, load the existing index
    if not create_anyway and os.path.exists(index_path):
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # Otherwise, load the PDF, split it into chunks, create a new FAISS index, and save it
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vectorstore.save_local(index_path)
    
    return vectorstore, splits

# Function to load or create a FAISS index used for routing queries to the correct year's data
def load_or_create_faiss_index_for_routing(years: list, embeddings: OpenAIEmbeddings,
                                           create_anyway: bool = False) -> FAISS:
    routing_index_path = os.path.join(FAISS_INDEX_DIR, "routing")
    
    # If index exists and recreate flag is false, load the routing index
    if not create_anyway and os.path.exists(routing_index_path):
        routing_vectorstore = FAISS.load_local(
            routing_index_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        # Otherwise, create a new FAISS index using year labels and save it
        routing_vectorstore = FAISS.from_texts(texts=years, embedding=embeddings)
        routing_vectorstore.save_local(routing_index_path)
    
    return routing_vectorstore

# Initialize data for each year: load documents, create indices, and prepare retrievers and QA tools
def initialize_year_data(years_dict: dict, chunk_size: int = 1000,
                         chunk_overlap: int = 200, recreate_indexes: bool = True):
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.chat_models import ChatOpenAI
    
    # Initialize embeddings and a language model
    embeddings = OpenAIEmbeddings(api_key=api_key)
    llm = ChatOpenAI(api_key=api_key, model='gpt-4')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Dictionaries to store data for each year
    year_docs = {}
    year_vectorstores = {}
    year_retrievers = {}
    year_qa_tools = {}

    # Loop over each year and process its documents
    for year, file_path in years_dict.items():
        # Load or create the FAISS index and retrieve document chunks
        vectorstore, splits = load_or_create_faiss_index_from_documents(
            year=year,
            file_path=file_path,
            embeddings=embeddings,
            text_splitter=text_splitter,
            create_anyway=recreate_indexes
        )
        
        # Create a retriever for the current year's vector store
        retriever = vectorstore.as_retriever()
        year_vectorstores[year] = vectorstore
        year_retrievers[year] = retriever
        year_docs[year] = splits
    
    return {
        "year_docs": year_docs,
        "year_vectorstores": year_vectorstores,
        "year_retrievers": year_retrievers,
        "year_qa_tools": year_qa_tools
    }

# Initialize embeddings and language model for later use
embeddings = OpenAIEmbeddings(api_key=api_key)
llm = ChatOpenAI(api_key=api_key, model='gpt-4')

# Initialize data for each year, including vectors and retrievers
data = initialize_year_data(years, recreate_indexes=False)
year_docs = data['year_docs']
year_vectorstores = data['year_vectorstores']
year_retrievers = data['year_retrievers']
year_qa_tools = data['year_qa_tools']

# Create or load the routing vectorstore to map queries to the correct year
routing_vectorstore = load_or_create_faiss_index_for_routing(
    years=list(years.keys()),
    embeddings=embeddings,
    create_anyway=False
)

# Function to find the most similar year to a given query using the routing vectorstore
def get_most_similar_year(query: str):
    try:
        # Search for the most similar year based on the query
        similar_docs = routing_vectorstore.similarity_search(query, k=1)
        if similar_docs:
            return similar_docs[0].page_content
        else:
            return None
    except Exception as e:
        return None

# Pydantic models for input validation for router and QA tools
class RouterInput(BaseModel):
    query: str = Field(..., description="The query to determine the relevant year.")

class QAInput(BaseModel):
    query: str = Field(..., description="The query for the QA tool.")

# Semantic router function that uses the similarity search to determine the year
def semantic_router(query: str) -> str:
    year = get_most_similar_year(query)
    # Check if the found year is one we have in our dataset
    if year and year in years:
        return year
    else:
        return "No Matching Year"

# Create a structured tool for routing queries to the correct year
router_tool = StructuredTool.from_function(
    func=semantic_router,
    name="Semantic Router",
    description=(
        "Determines which year's knowledge base to use for answering the question. "
        "Requires 'query' as input and returns the year or 'No Matching Year' if no match."
    ),
    args_schema=RouterInput
)

# Function to run the QA chain for a specific year and calculate readability score of the answer
def run_qa(chain: RetrievalQA, query: str) -> str:
    try:
        # Execute the QA chain to get an answer from documents
        response = chain.run(query)
        # Compute Flesch-Kincaid grade for the response text
        fk_grade = textstat.flesch_kincaid_grade(response)
        # Append readability info to the answer
        return f"{response}\n\nFlesch-Kincaid Grade: {fk_grade}"
    except Exception as e:
        return "I'm sorry, I encountered an error while processing your request."

# Create QA tools for each year using the appropriate retrievers
qa_tools = []
for year, retriever in year_retrievers.items():
    # Set up a RetrievalQA chain for the current year's documents
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    # Partially bind the QA chain to the run_qa function
    bound_qa_run = partial(run_qa, chain=qa_chain)
    # Create a structured tool for the current year's QA
    qa_tool = StructuredTool.from_function(
        func=bound_qa_run,
        name=f"{year} QA",
        description=(
            f"Use this tool to answer questions about {year}'s information, "
            "especially agricultural information."
        ),
        args_schema=QAInput
    )
    
    qa_tools.append(qa_tool)

# Combine router tool and all year-specific QA tools into one toolset for the agent
tools = [router_tool] + qa_tools

# Define system instructions for the agent to follow
system_message = """
You are an assistant that helps answer agricultural questions based on UNL research reports.

When given a question, you should:
1. Use the 'Semantic Router' tool to determine which year's information is relevant to the question (UNL-Agriculture-2013 to UNL-Agriculture-2023).
2. Use the appropriate '{year} QA' tool to find information from the selected year's documents.
3. When answering, consider any previous relevant queries asked by the user and mention if they are helpful.
"""

# Agent configuration arguments
agent_kwargs = {
    "system_message": system_message
}

# Initialize conversation memory to keep track of previous queries/responses
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=7)

# Initialize the agent with tools, language model, system message, and memory
agent_chain = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory
)

# Flask route for chat endpoint that receives user queries and returns answers
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    # Validate input
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request"}), 400
    
    user_query = data['query']
    try:
        # Run the agent chain with the user query to get a response
        answer = agent_chain.run(user_query)
        return jsonify({"answer": answer})
    except Exception as e:
        # Handle unexpected errors gracefully
        return jsonify({"error": str(e)}), 500

# Optional: Basic route to check if the API is running
@app.route('/', methods=['GET'])
def home():
    return "LangChain Flask API is running."

# Entry point to run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
