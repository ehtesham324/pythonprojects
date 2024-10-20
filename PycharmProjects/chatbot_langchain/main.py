from dotenv import load_dotenv
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Retrieve the API key from the environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found. Make sure it's set in the .env file.")

# Initialize the LLM with the API key
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)

# Initialize memory in session state if it doesn't exist
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# Use the memory stored in session state
memory = st.session_state.memory

# Set up conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# Initialize message history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Chatbot")

# User input
user_input = st.chat_input("Type something")

if user_input:
    # Append the user's message to the session state messages
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate the response using the conversation chain
    response = conversation.predict(input=user_input)

    # Append the bot's response to the session state messages
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display the conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Debug: Display the memory buffer to verify conversation history
# st.write("**Memory Buffer:**", memory.load_memory_variables({}))
# ---------------------------------------------------------------------------------
# from dotenv import load_dotenv
# import os
# import streamlit as st
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
#
# # Load environment variables
# load_dotenv()
#
# # Retrieve the API key from the environment
# api_key = os.getenv("GOOGLE_API_KEY")
# if not api_key:
#     raise ValueError("API key not found. Make sure it's set in the .env file.")
#
# # Initialize the LLM with the API key
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)
#
# # Initialize memory in session state if it doesn't exist
# if 'memory' not in st.session_state:
#     st.session_state.memory = ConversationBufferMemory()
#
# # Use the memory stored in session state
# memory = st.session_state.memory
#
# # Set up conversation chain with memory
# conversation = ConversationChain(
#     llm=llm,
#     memory=memory,
#     verbose=False
# )
#
# # Initialize message history in session state if it doesn't exist
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#
# # Custom CSS for a more appealing UI
# st.markdown("""
#     <style>
#     .chat-container {
#         max-width: 700px;
#         margin: 0 auto;
#         padding: 10px;
#     }
#     .chat-bubble {
#         padding: 10px 15px;
#         border-radius: 10px;
#         margin-bottom: 10px;
#         max-width: 70%;
#         word-wrap: break-word;
#     }
#     .chat-bubble-user {
#         background-color: #DCF8C6;
#         align-self: flex-start;
#         color: black;
#     }
#     .chat-bubble-bot {
#         background-color: #F1F1F1;
#         align-self: flex-end;
#         color: black;
#     }
#     .chat-box {
#         display: flex;
#         justify-content: flex-end;
#         flex-direction: column;
#         align-items: flex-start;
#     }
#     .chat-box-bot {
#         align-items: flex-end;
#     }
#     .chat-input {
#         margin-top: 10px;
#         width: 100%;
#     }
#     </style>
# """, unsafe_allow_html=True)
#
# st.title("🌟 Chatbot")
#
# # User input
# user_input = st.chat_input("Type something...")
#
# if user_input:
#     # Append the user's message to the session state messages
#     st.session_state.messages.append({"role": "user", "content": user_input})
#
#     # Generate the response using the conversation chain
#     response = conversation.predict(input=user_input)
#
#     # Append the bot's response to the session state messages
#     st.session_state.messages.append({"role": "assistant", "content": response})
#
# # Display the conversation history
# st.markdown('<div class="chat-container">', unsafe_allow_html=True)
# for message in st.session_state.messages:
#     if message["role"] == "user":
#         st.markdown(f'<div class="chat-box chat-bubble chat-bubble-user">{message["content"]}</div>', unsafe_allow_html=True)
#     else:
#         st.markdown(f'<div class="chat-box chat-box-bot chat-bubble chat-bubble-bot">{message["content"]}</div>', unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)
#

# -------------------------------------------------------------------------------------
# from dotenv import load_dotenv
# import os
# import streamlit as st
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import ConversationChain
# from langchain.chains.router import MultiPromptChain
# from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
# from langchain.prompts import PromptTemplate
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains import LLMChain
#
# load_dotenv()
#
# api_key = os.getenv("GOOGLE_API_KEY")
# if not api_key:
#     raise ValueError("API key not found. Make sure it's set in the .env file.")
#
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)
#
# physics_template = """You are a very smart physics professor. \
# You are great at answering questions about physics in a concise\
# and easy to understand manner. \
# When you don't know the answer to a question you admit\
# that you don't know.
#
# Here is a question:
# {input}"""
#
#
# math_template = """You are a very good mathematician. \
# You are great at answering math questions. \
# You are so good because you are able to break down \
# hard problems into their component parts,
# answer the component parts, and then put them together\
# to answer the broader question.
#
# Here is a question:
# {input}"""
#
# history_template = """You are a very good historian. \
# You have an excellent knowledge of and understanding of people,\
# events and contexts from a range of historical periods. \
# You have the ability to think, reflect, debate, discuss and \
# evaluate the past. You have a respect for historical evidence\
# and the ability to make use of it to support your explanations \
# and judgements.
#
# Here is a question:
# {input}"""
#
#
# computerscience_template = """ You are a successful computer scientist.\
# You have a passion for creativity, collaboration,\
# forward-thinking, confidence, strong problem-solving capabilities,\
# understanding of theories and algorithms, and excellent communication \
# skills. You are great at answering coding questions. \
# You are so good because you know how to solve a problem by \
# describing the solution in imperative steps \
# that a machine can easily interpret and you know how to \
# choose a solution that has a good balance between \
# time complexity and space complexity.
#
# Here is a question:
# {input}"""
#
# prompt_infos = [
#     {
#         "name": "physics",
#         "description": "Good for answering questions about physics",
#         "prompt_template": physics_template
#     },
#     {
#         "name": "math",
#         "description": "Good for answering math questions",
#         "prompt_template": math_template
#     },
#     {
#         "name": "History",
#         "description": "Good for answering history questions",
#         "prompt_template": history_template
#     },
#     {
#         "name": "computer science",
#         "description": "Good for answering computer science questions",
#         "prompt_template": computerscience_template
#     }
# ]
#
# destination_chains = {}
# for p_info in prompt_infos:
#     name = p_info["name"]
#     prompt_template = p_info["prompt_template"]
#     prompt = ChatPromptTemplate.from_template(template=prompt_template)
#     chain = LLMChain(llm=llm, prompt=prompt)
#     destination_chains[name] = chain
#
# destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
# destinations_str = "\n".join(destinations)
#
# default_prompt = ChatPromptTemplate.from_template("{input}")
# default_chain = LLMChain(llm=llm, prompt=default_prompt)
#
# MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
# language model select the model prompt best suited for the input. \
# You will be given the names of the available prompts and a \
# description of what the prompt is best suited for. \
# You may also revise the original input if you think that revising\
# it will ultimately lead to a better response from the language model.
#
# << FORMATTING >>
# Return a markdown code snippet with a JSON object formatted to look like:
# ```json
# {{{{
#     "destination": string \ name of the prompt to use or "DEFAULT"
#     "next_inputs": string \ a potentially modified version of the original input
# }}}}
# ```
#
# REMEMBER: "destination" MUST be one of the candidate prompt \
# names specified below OR it can be "DEFAULT" if the input is not\
# well suited for any of the candidate prompts.
# REMEMBER: "next_inputs" can just be the original input \
# if you don't think any modifications are needed.
#
# << CANDIDATE PROMPTS >>
# {destinations}
#
# << INPUT >>
# {{input}}
#
# << OUTPUT (remember to include the ```json)>>"""
#
# router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
#     destinations=destinations_str
# )
# router_prompt = PromptTemplate(
#     template=router_template,
#     input_variables=["input"],
#     output_parser=RouterOutputParser(),
# )
#
# router_chain = LLMRouterChain.from_llm(llm, router_prompt)
#
# chain = MultiPromptChain(router_chain=router_chain,
#                          destination_chains=destination_chains,
#                          default_chain=default_chain, verbose=True
#                         )