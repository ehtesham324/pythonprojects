# pet name generator
from dotenv import load_dotenv
import os
import google.generativeai as genai
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found. Make sure it's set in the .env file.")

# Configure Google Generative AI with the API key
genai.configure(api_key=api_key)

# Initialize the LLM with the API key passed as a named parameter
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)

st.title("Pet Name Generator")

type_of_animal = st.sidebar.text_input("Type of animal")
colour = st.sidebar.text_input("Colour of your animal")

if st.sidebar.button("Enter"):
    template_string = """You are given the animal \
    that is delimited by triple backticks \
    and its colour {colour}. Give 5 names that can be used as pet names \
    and remember while generating response do not number the names use bullets instead\
    animal: ```{animal}```
    """

    prompt_template = ChatPromptTemplate.from_template(template_string)
    customer_messages = prompt_template.format_messages(
        animal=type_of_animal,
        colour=colour
    )

    customer_response = llm(customer_messages)

# Displaying the response in Streamlit

    st.write(customer_response.content)
# ------------------------------------------------------------------------------------


