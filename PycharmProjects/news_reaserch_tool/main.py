import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from utils import clean_text
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found. Make sure it's set in the .env file.")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)

# Set the title of the Streamlit app
st.title("News Research Tool")

# Define user inputs
url_input = st.sidebar.text_input("Enter a URL:", value="https://jobs.nike.com/job/R-31388?from=job%20search%20funnel")
fetch_and_process = st.sidebar.button("Fetch and Process")
query = st.text_input("Enter your question")
submit = st.button("Submit")

# Persistent directory for storing the Chroma database
persist_directory = "chroma_db"

# Initialize variables
if 'db' not in st.session_state:GOOGLE_API_KEY
    st.session_state.db = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

# Fetch and process the URL to store in persistent Chroma DB
if fetch_and_process:
    try:
        loader = WebBaseLoader([url_input])
        documents = loader.load()

        if not documents:
            st.error("No content loaded from the provided URL.")
        else:
            text = str(documents[0])

            # Clean the text
            data = clean_text(text)

            # Split the text into chunks
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=40)
            split_texts = text_splitter.split_text(data)

            # Create embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            # Create a persistent Chroma database and store the data
            db = Chroma.from_texts(split_texts, embeddings, persist_directory=persist_directory)

            # Persist the embeddings and db in session state
            st.session_state.db = db
            st.session_state.embeddings = embeddings


            st.success("Data processed and stored in the persistent database.")
    except Exception as e:
        st.error(f"An Error Occurred: {e}")

# Query processing and similarity search
if submit:
    if st.session_state.db is None:
        # If database is not in memory, load from the persistent directory
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

            # Store in session state
            st.session_state.db = db
            st.session_state.embeddings = embeddings
        except Exception as e:
            st.error(f"Error loading the database: {e}")

    if st.session_state.db is not None:
        try:
            # Embed the query
            query_embedding = st.session_state.embeddings.embed_query(query)

            # Perform similarity search using the retriever
            search_results = st.session_state.db.similarity_search_by_vector(query_embedding)

            if search_results:
                # Create a prompt template with the search results
                first_prompt = ChatPromptTemplate.from_template(
                    f"""You will act as a news research tool. 
                    You are provided with the query of the user: {query}.
                    And relevant text extracted from the news site: 
                    {search_results}  
                    You will answer the question of the user using the provided text.
                    No preamble."""
                )

                # Chain to handle query processing
                chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="answer")

                # Sequential chain for combining everything
                overall_chain = SequentialChain(
                    chains=[chain_one],
                    input_variables=["query", "search_results"],
                    output_variables=["answer"],
                    verbose=False
                )

                # Get the final result
                result = overall_chain({"query": query, "search_results": search_results})

                # Display the result
                st.write(f"**Answer:** {result['answer']}")
            else:
                st.error("No similar documents found.")
        except Exception as e:
            st.error(f"An Error Occurred: {e}")

