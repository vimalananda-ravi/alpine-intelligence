import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
import os
from pinecone import Pinecone, ServerlessSpec
import time
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score
# from sqlalchemy import create_engine, Column, String, Integer, Text
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
import random 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.graph_objects as go

nltk.download('vader_lexicon')


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# Streamlit page configuration with a wide layout
st.set_page_config(page_title="Alpine Intelligence", layout="wide", page_icon="https://cdn.icon-icons.com/icons2/2699/PNG/512/mulesoft_logo_icon_170933.png")


# Initialize dark mode state
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

# Dark mode toggle
dark_mode = st.sidebar.checkbox("🌓 Toggle Dark Mode", value=st.session_state["dark_mode"])
st.session_state["dark_mode"] = dark_mode

# Add custom CSS for background color and conversational layout
if dark_mode:
    st.markdown(
        """
        <style>
        /* Dark mode styles */
        .stApp {
            background-color: #1e1e1e;
            color: white;
        }
        input {
            border: 2px solid #055289;
            border-radius: 10px;
            padding: 10px;
            color: white;
            background-color: #333;
            font-family: 'Calibri', sans-serif;
        }
        button {
            background-color: #055289;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            border-radius: 10px;
            font-family: 'Calibri', sans-serif;
        }
        .sidebar .sidebar-content {
            padding: 20px;
            color: white;
        }
        h1, h2, h3 {
            color: #055289;
        }
        .chatbox {
            background-color: #2a2a2a;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(255, 255, 255, 0.1);
            font-family: 'Calibri', sans-serif;
        }
        </style>
        """, unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        /* Light mode styles */
        .stApp {
            background-color: white;
            color: black;
        }
        input {
            border: 2px solid #055289;
            border-radius: 10px;
            padding: 10px;
            color: #333;
        }
        button {
            background-color: #055289;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            border-radius: 10px;
        }
        .sidebar .sidebar-content {
            padding: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #00A1DF;
        }
        .chatbox {
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        </style>
        """, unsafe_allow_html=True
    )

# Sidebar content
st.sidebar.title("🦜KnowledgeHub Agent")
st.sidebar.write("Ask questions on MuleSoft best practices, and the agent will provide answers based on Knowledge Hub posts.")


# AI Insights Dashboard in the sidebar
st.sidebar.header("📊 AI Insights Dashboard")


total_queries = 0
avg_response_time = 0.0
sentiment_label = "N/A"
compound_score = 0.0


if "chat_history" in st.session_state:
   total_queries = len(st.session_state["chat_history"]) // 2  # Each query has a response
   avg_response_time = round(random.uniform(1, 2), 2)  # Simulating average response time
   sentiment_analyzer = SentimentIntensityAnalyzer()


   # Sentiment analysis
   if "sentiments" not in st.session_state:
       st.session_state["sentiments"] = []


   # Analyze the sentiment of the user's messages
   for i, (speaker, msg) in enumerate(st.session_state["chat_history"]):
       if speaker == "You":  # Only analyze user queries
           sentiment_score = sentiment_analyzer.polarity_scores(msg)
           st.session_state["sentiments"].append(sentiment_score)


   # Display sentiment score for the last user message
   if st.session_state["sentiments"]:
       last_sentiment = st.session_state["sentiments"][-1]
       compound_score = last_sentiment['compound']


       # Determine if the sentiment is Positive, Negative, or Neutral
       if compound_score > 0:
           sentiment_label = "Positive"
       elif compound_score < 0:
           sentiment_label = "Negative"
       else:
           sentiment_label = "Neutral"


# Show some insights on the sidebar
st.sidebar.metric("Total Queries", total_queries)
st.sidebar.metric("Avg Response Time (s)", avg_response_time)
st.sidebar.subheader("Sentiment Analysis")
st.sidebar.write(f"Sentiment of last user message: {sentiment_label} (Score: {compound_score})")



# Main Title
# Create two columns
col1, col2 = st.columns([1, 14])  # Adjust the ratio for logo and title sizes

# Add the image in the first column
with col1:
    st.image("https://cdn.icon-icons.com/icons2/2699/PNG/512/mulesoft_logo_icon_170933.png", width=75)  # Replace with your image file

# Add the title in the second column
with col2:
    st.title("Alpine Intelligence")

st.write("Welcome to the MuleSoft KnowledgeHub Agent! Ask questions about best practices, operations, scalability and more.")

# Setup API tokens (hide API keys in real applications)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["PINECONE_API_KEY"] = "c3e3268e-177d-4779-a1e9-8142ad1b6b9d"
os.environ["PINECONE_ENV"] = "us-east-1"

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "alpine-intelligence"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=os.environ["PINECONE_ENV"])
    )

# Initialize the embedding model
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/intfloat/multilingual-e5-large")

# Initialize Langchain Pinecone Vector Store
vectorstore = LangchainPinecone.from_existing_index(index_name=index_name, embedding=embedding_model)

# Initialize the LLM
llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-Small-Instruct-2409")

# Initialize the RetrievalQA chain with embeddings and LLM
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# Chat history list
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# Layout for chatbot and dashboard
col1, col2 = st.columns([3, 1])

# Chatbot section (Left side)
with col1:
    # User query input
    user_query = st.text_input("Enter your  question:")

    # Process the query when entered
    if user_query:
        instruction = ("You are an answering agent responsible for synthesizing relevant information from multiple sources to answer the user’s query. Your task is to analyze the provided text segments and construct a concise, coherent response based on the original question. Analyze the provided text segments.Identify key pieces of information that are relevant to the original query.Summarize the information from different segments where applicable, avoiding redundancy.Ensure that the response is concise, coherent, and directly answers the user’s original query.The final response should be well-structured and contain no irrelevant or extraneous information.Output the final response in a paragraph format.")
        query_with_instruction = instruction + " " + user_query

        # Display a spinner while processing
        with st.spinner('Processing your query...'):
            start_time = time.time()
            response = qa_chain.run(query=query_with_instruction, max_length=500, return_only_outputs=True)
            end_time = time.time()
            response_time = end_time - start_time

        # Extract the helpful answer
        if isinstance(response, str):
            helpful_answer = response.split("Helpful Answer:")[-1].strip()
        else:
            helpful_answer = response.get("output", "No helpful answer found").strip()

        # Append the conversation to chat history
        st.session_state["chat_history"].append(("You", user_query))
        st.session_state["chat_history"].append(("AlpineIntelligence", helpful_answer))



# Display chat history in the main area
for speaker, msg in st.session_state["chat_history"]:
    st.markdown(f"<div class='chatbox'><strong>{speaker}:</strong> {msg}</div>", unsafe_allow_html=True)
