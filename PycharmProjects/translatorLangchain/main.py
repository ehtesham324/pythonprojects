# Translator
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

st.title("Translator")
text = st.text_input("Enter the text to translate")
language_in = st.text_input("Language of your text")
language_to = st.text_input("Language to translate text")
if st.button("Enter"):
    template_string = f"""You will act as an translator \
    you are provided with the text \
    that is delimited by triple backticks \
    and the language it is in {language_in}.\
    translate the text in {language_to}
    text to translate: ```{text}```
    """

    prompt_template = ChatPromptTemplate.from_template(template_string)
    customer_messages = prompt_template.format_messages(
        language_in=language_in,
        language_to=language_to,
        text = text
    )

    customer_response = llm(customer_messages)

# Displaying the response in Streamlit

    st.write(customer_response.content)