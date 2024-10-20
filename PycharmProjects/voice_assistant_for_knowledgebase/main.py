
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_chroma import Chroma
# from utils import clean_text
# from audio_recorder_streamlit import audio_recorder
# from langchain.chains import LLMChain
# from langchain.chains import SequentialChain
# from langchain.chains import RetrievalQA
# ------------------------------------------------------------------------------------
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from utils import clean_text
from audio_recorder_streamlit import audio_recorder
from groq import Groq
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Initialize the Groq client
client = Groq()

# Load API keys
api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
if not api_key or not groq_api_key:
    raise ValueError("API key not found. Make sure it's set in the .env file.")

# Initialize LLMs
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)

# Persistent directory for storing the Chroma database
persist_directory = "chroma_db"


# Helper functions to load data and embed
def load_scrap_store():
    """Loads and processes data from URLs and stores it in a persistent Chroma DB."""
    url_input = [
        'https://huggingface.co/docs/huggingface_hub/guides/overview',
        'https://huggingface.co/docs/huggingface_hub/guides/download'
    ]
    try:
        loader = WebBaseLoader(url_input)
        documents = loader.load()

        if not documents:
            st.error("No content loaded from the provided URL.")
        else:
            text = str(documents[0])
            data = clean_text(text)

            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=40)
            split_texts = text_splitter.split_text(data)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            db = Chroma.from_texts(split_texts, embeddings, persist_directory=persist_directory)

            st.session_state.db = db
            st.success("Data processed and stored in the persistent database.")
    except Exception as e:
        st.error(f"An Error Occurred: {e}")


# Load existing DB as retriever
def load_db():
    if "db" not in st.session_state:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        st.session_state.db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


# Transcribe audio using Google Cloud Speech-to-Text
def transcribe_audio(audio_file_path):
    with open(audio_file_path, "rb") as file:
        # Create a transcription of the audio file
        transcription = client.audio.transcriptions.create(
            file=(audio_file_path, file.read()),  # Required audio file
            model="distil-whisper-large-v3-en",  # Required model to use for transcription
            prompt="Specify context or spelling",  # Optional
            response_format="json",  # Optional
            language="en",  # Optional
            temperature=0.0  # Optional
        )
        # Print the transcription text
        transcription = transcription.text

    return transcription


# Record and transcribe audio
def record_and_transcribe_audio():
    """Records audio and transcribes it into text."""
    audio_bytes = audio_recorder()
    transcription = None
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)

        if st.button("Transcribe"):
            transcription = transcribe_audio("temp_audio.wav")
            os.remove("temp_audio.wav")
            st.write(f"Transcription: {transcription}")

    return transcription


# Perform similarity search using Chroma DB
def search_db(query):
    """Performs similarity search on the Chroma DB based on a query embedding."""
    if "db" not in st.session_state:
        st.error("Please load the Chroma database first.")
        return None
    try:
        # Embed the query using the GoogleGenerativeAIEmbeddings directly
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        query_embedding = embeddings.embed_query(query)

        # Perform similarity search with the embedded query
        search_results = st.session_state.db.similarity_search_by_vector(query_embedding)
        return search_results
    except Exception as e:
        st.error(f"Error during similarity search: {e}")
        return None


def llm_response(query, context):
    llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    # Ensure both query and context are strings
    query = str(query)
    context = str(context)

    prompt_template = """
    ### INSTRUCTION:
    You are provided with a user query: {query} and 
    the text from which you have to answer: {context}
    Do not provide a preamble.
    """
    # Format the prompt with the user query and context
    prompt = prompt_template.format(query=query, context=context)

    # Get response from LLM
    response = llm.invoke(prompt)

    return response.content


# Main function to run the app
def main():
    # Load the Chroma DB
    load_db()

    # Initialize Streamlit app
    st.title("Voice Assistant ")

    # Record and transcribe audio
    transcription = record_and_transcribe_audio()

    # Get user input from the transcribed audio or a text input box
    if transcription:
        user_input = transcription
    else:
        user_input = st.text_input("Or enter text query manually:")

    # Perform a search on the database if there's user input
    if user_input:
        search_results = search_db(user_input)

        if search_results:
            # Combine the search results into a single context string
            context = "\n".join([str(result.page_content) for result in search_results])

            # Get the LLM response based on the query and the context
            response = llm_response(user_input, context)
            st.write(response)
        else:
            st.write("No relevant documents found.")


# Run the main function
if __name__ == "__main__":
    main()
