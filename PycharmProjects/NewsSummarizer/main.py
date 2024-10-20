# #NEWS SUMMARIZER
# from dotenv import load_dotenv
# import os
# import google.generativeai as genai
# import streamlit as st
# from langchain_core.messages import HumanMessage
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import ChatPromptTemplate
# import requests
#
# # Load environment variables from .env file
# load_dotenv()
#
# # Retrieve the API key from the environment
# api_key = os.getenv("GOOGLE_API_KEY")
# if not api_key:
#     raise ValueError("API key not found. Make sure it's set in the .env file.")
#
# # Configure Google Generative AI with the API key
# genai.configure(api_key=api_key)
#
# # Initialize the LLM with the API key passed as a named parameter
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)
#
# news_api_key = os.getenv("NEWS_API_KEY")
#
#
# def fetch_articles(keyword):
#     url = f"https://newsapi.org/v2/everything?q={keyword}&apiKey={news_api_key}"
#     response = requests.get(url)
#     articles = response.json().get('articles', [])
#     return articles
#
# def summarize(content):
#     template_string = """You will act as a news summarizer./
#             You are provided with the news {content}./
#             You have to provide its summary and the category of the news article.
#             """
#
#     # Create the prompt
#     prompt_template = ChatPromptTemplate.from_template(template_string)
#     customer_messages = prompt_template.format_messages(
#         content=content
#     )
#
#     customer_response = llm(customer_messages)
#
#     st.write("Summarized News:")
#     st.write(customer_response)
# st.title("News Summarizer")
# keyword = st.sidebar.text_input("Enter the keyword to get news articles")
# if st.sidebar.button("Get News and Summarize"):
#     if keyword:
#         articles = fetch_articles(keyword)
#         if articles:
#             st.write(f'Found {len(articles)} on {keyword}')
#             st.write('Top 5 articles')
#             for article in articles[:5]:
#                 title = article['title']
#                 content = article['content']
#                 source = article['source', {}]
#                 url = article['url']
#                 if content:
#                     st.subheader(title)
#                     st.write(f"**Source:** {source}")
#                     summarize(content)
#
#         else:
#             st.write('No articles found!')
#     else:
#         st.write("No keyword entered")
#
#
#
# # NEWS SUMMARIZER
# from dotenv import load_dotenv
# import os
# import google.generativeai as genai
# import streamlit as st
# from langchain_core.messages import HumanMessage
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import ChatPromptTemplate
# import requests
#
# # Load environment variables from .env file
# load_dotenv()
#
# # Retrieve the API key from the environment
# api_key = os.getenv("GOOGLE_API_KEY")
# if not api_key:
#     raise ValueError("API key not found. Make sure it's set in the .env file.")
#
# # Configure Google Generative AI with the API key
# genai.configure(api_key=api_key)
#
# # Initialize the LLM with the API key passed as a named parameter
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)
#
# news_api_key = os.getenv("NEWS_API_KEY")
# if not news_api_key:
#     raise ValueError("NewsAPI key not found. Make sure it's set in the .env file.")
#
#
# def fetch_articles(keyword):
#     url = f"https://newsapi.org/v2/everything?q={keyword}&apiKey={news_api_key}"
#     response = requests.get(url)
#     articles = response.json().get('articles', [])
#     return articles


# def summarize(content):
#     template_string = """You will act as a news summarizer.
#     You are provided with the news article: {content}.
#     You have to provide its summary and the category of the news article."""
#
#     # Create the prompt
#     prompt_template = ChatPromptTemplate.from_template(template_string)
#     customer_messages = prompt_template.format_messages(
#         content=content
#     )
#
#     customer_response = llm(customer_messages)
#
#     st.write(customer_response.content)
#
#
# st.title("News Summarizer")
# keyword = st.sidebar.text_input("Enter the keyword to get news articles")
# if st.sidebar.button("Get News and Summarize"):
#     if keyword:
#         articles = fetch_articles(keyword)
#         if articles:
#             st.write(f'Found {len(articles)} articles on "{keyword}".')
#             st.write('Top 5 articles:')
#             for article in articles[:5]:
#                 title = article['title']
#                 content = article['content']
#                 source = article.get('source', {}).get('name', 'Unknown Source')
#                 url = article['url']
#
#                 if content:
#                     st.subheader(title)
#                     st.write(f"**Source:** {source}")
#                     st.write(f"[Read more]({url})")
#                     summarize(content)
#                 else:
#                     st.subheader(title)
#                     st.write(f"**Source:** {source}")
#                     st.write(f"[Read more]({url})")
#                     st.write("No content available for summarization.")
#         else:
#             st.write('No articles found!')
#     else:
#         st.write("Please enter a keyword.")
# ----------------------------------------------------------------------------
# # NEWS SUMMARIZER using langchain sequential chains
from dotenv import load_dotenv
import os
import google.generativeai as genai
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import requests
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

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

news_api_key = os.getenv("NEWS_API_KEY")
if not news_api_key:
    raise ValueError("NewsAPI key not found. Make sure it's set in the .env file.")


def fetch_articles(keyword):
    url = f"https://newsapi.org/v2/everything?q={keyword}&apiKey={news_api_key}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return articles


def summarize(content):

    # Chain 1: Generate summary from content
    first_prompt = ChatPromptTemplate.from_template(
    """You will act as a news summarizer. 
    You are provided with the news article: {content}.  
    Provide a summary of the news article."""
    )
    chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="summary")

    # Chain 2: Determine the category of the news based on the summary
    second_prompt = ChatPromptTemplate.from_template(
        """Based on the summary, provide the category for the news: {summary}"""
    )
    chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="news_category")

    # Overall chain that takes content and produces a summary and category
    overall_chain = SequentialChain(
        chains=[chain_one, chain_two],
        input_variables=["content"],
        output_variables=["summary", "news_category"],
        verbose=True
    )

    result = overall_chain({"content": content})

    return result


st.title("News Summarizer")
keyword = st.sidebar.text_input("Enter the keyword to get news articles")
if st.sidebar.button("Get News and Summarize"):
    if keyword:
        articles = fetch_articles(keyword)
        if articles:
            st.write(f'Found {len(articles)} articles on "{keyword}".')
            st.write('Top 5 articles:')
            for article in articles[:5]:
                title = article['title']
                content = article['content']
                source = article.get('source', {}).get('name', 'Unknown Source')
                url = article['url']

                if content:
                    st.subheader(title)
                    st.write(f"**Source:** {source}")
                    st.write(f"[Read more]({url})")

                    # Generate summary and category
                    result = summarize(content)
                    st.write(f"**Summary:** {result['summary']}")
                    st.write(f"**Category:** {result['news_category']}")
                else:
                    st.subheader(title)
                    st.write(f"**Source:** {source}")
                    st.write(f"[Read more]({url})")
                    st.write("No content available for summarization.")
        else:
            st.write('No articles found!')
    else:
        st.write("Please enter a keyword.")
