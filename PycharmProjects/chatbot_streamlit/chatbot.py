# import streamlit as st
# from dotenv import load_dotenv
# import os
# import google.generativeai as genai
#
# load_dotenv()
#
#
# api_key = os.getenv("api_key")
# if not api_key:
#     raise ValueError("API key not found. Make sure it's set in the .env file.")
#
# genai.configure(api_key=api_key)
# model = genai.GenerativeModel('gemini-1.5-flash')
#
# conversation_history = []
#
# def generate_response(user_input):
#
#     conversation_history.append(f"User: {user_input}")
#
#     history_string = "\n".join(conversation_history)
#
#     response = model.generate_content(history_string)
#
#     ai_response = response.text if hasattr(response, 'text') else "Error: Response format unexpected."
#
#     conversation_history.append(f"AI: {ai_response}")
#
#     return ai_response
#
# def main():
#     st.title("Chatbot")
#
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#
#     # Input area
#     # creates form for input
#     with st.form(key='chat_form', clear_on_submit=True):
#         # creates text input field
#         user_input = st.text_input("You:", key="user_input")
#         # creates submit button
#         submitted = st.form_submit_button("Send")
#
#     if submitted and user_input:
#         # Generate the response
#         ai_response = generate_response(user_input)
#
#         # Append the messages to session state
#         st.session_state.messages.append({"role": "user", "content": user_input})
#         st.session_state.messages.append({"role": "assistant", "content": ai_response})
#
#     # Display chat history from the top
#     for message in st.session_state.messages:
#         if message["role"] == "user":
#             st.markdown(f"**You:** {message['content']}")
#         else:
#             st.markdown(f"**AI:** {message['content']}")
#
#     # Button to clear the conversation
#     if st.button("Clear Conversation"):
#         st.session_state.messages = []
#         conversation_history.clear()  # Also clear the local conversation history
#
# if __name__ == "__main__":
#     main()
#--------------------------------------------------------------------------------------
# import streamlit as st
# from dotenv import load_dotenv
# import os
# import google.generativeai as genai
#
# load_dotenv()
#
# api_key = os.getenv("api_key")
# if not api_key:
#     raise ValueError("API key not found. Make sure it's set in the .env file.")
#
# genai.configure(api_key=api_key)
# model = genai.GenerativeModel('gemini-1.5-flash')
#
#
# def generate_response(user_input):
#     try:
#         response = model.generate_content(user_input)
#         return response.text
#     except Exception as e:
#         return f"Error: {e}"
#
#
# def main():
#     st.title("Chatbot")
#
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#
#     # Display chat messages from history on app rerun
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#
#     # Accept user input
#     if prompt := st.chat_input("What is up?"):
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         # Display user message in chat message container
#         with st.chat_message("user"):
#             st.markdown(prompt)
#         with st.chat_message("assistant"):
#             history_string = "\n".join([msg["content"] for msg in st.session_state.messages])
#             response = generate_response(history_string)
#             st.markdown(response)
#         st.session_state.messages.append({"role": "assistant", "content": response})
#
#     # Button to clear the conversation
#     if st.button("Clear Conversation"):
#         st.session_state.messages = []
#
#
# if __name__ == "__main__":
#     main()
    # ---------------------------------------------------------------------------------

# install ollama in computer
# download the model and install in terminal by ollama pull (model_name)
# in python file
# then  pip install langchain
# pip install langchain_ollama
# from langchain_ollama import OllamaLLM
# response = model.invoke("prompt")

# import streamlit as st
# from dotenv import load_dotenv
# import os
# import google.generativeai as genai
# import json
# import base64
#
# load_dotenv()
#
# api_key = os.getenv("api_key")
# if not api_key:
#     raise ValueError("API key not found. Make sure it's set in the .env file.")
#
# genai.configure(api_key=api_key)
# model = genai.GenerativeModel('gemini-1.5-flash')
#
# # --- User Preferences ---
# def load_user_prefs():
#     """Load user preferences from a JSON file."""
#     try:
#         with open("user_prefs.json", "r") as f:
#             return json.load(f)
#     except FileNotFoundError:
#         return {"background_color": "#333333", "chat_style": "default"}
#
# def save_user_prefs(prefs):
#     """Save user preferences to a JSON file."""
#     with open("user_prefs.json", "w") as f:
#         json.dump(prefs, f)
#
# # --- Response Generation ---
# def generate_response(user_input, context):
#     """Generate a response from the model, using the provided context."""
#     try:
#         response = model.generate_content(user_input)
#         return response.text
#     except Exception as e:
#         return f"Error: {e}"
#
# # --- Image Handling ---
# def handle_image_upload(image_file):
#     """Encode an uploaded image as base64 for sending to the model."""
#     if image_file:
#         image_bytes = image_file.read()
#         encoded_image = base64.b64encode(image_bytes).decode("utf-8")
#         return encoded_image
#     else:
#         return None
#
# # --- UI Functions ---
# def display_chat_history(messages):
#     """Display the chat history in the Streamlit interface."""
#     for message in messages:
#         with st.chat_message(message["role"]):
#             if "image" in message:
#                 st.image(message["image"])
#             else:
#                 st.markdown(message["content"])
#
# def clear_conversation():
#     """Clear the chat history and user input."""
#     st.session_state.messages = []
#
# # --- Streamlit App ---
# def main():
#     st.title("Chatbot")
#
#     user_prefs = load_user_prefs()
#
#     # --- Styling ---
#     st.markdown(
#         f"""
#         <style>
#         body {{
#             background-color: {user_prefs["background_color"]};
#             color: white;
#         }}
#         .stTextInput > div > div {{
#             background-color: #333333;
#             color: white;
#         }}
#         .stButton > button {{
#             background-color: #444444;
#             color: white;
#         }}
#         .st-chat-message {{
#             background-color: #444444;
#             border-radius: 10px;
#             padding: 10px;
#             margin: 5px 0;
#             color: white;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
#
#     # --- Chat History ---
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     display_chat_history(st.session_state.messages)

#     # --- User Input ---
#     prompt = st.chat_input("What is up?")
#
#     # --- Image Input ---
#     uploaded_image = st.file_uploader("Upload an image (optional)")
#     encoded_image = handle_image_upload(uploaded_image)
#
#     # --- Response Generation ---
#     if prompt:
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)
#         if encoded_image:
#             st.session_state.messages.append({"role": "user", "image": encoded_image})
#             with st.chat_message("user"):
#                 st.image(encoded_image)
#
#         # --- Context for Response ---
#         context = "\n".join(
#             [msg["content"] for msg in st.session_state.messages if "content" in msg]
#         )
#         if encoded_image:
#             context += f"\n\n Image: {encoded_image}"
#
#         response = generate_response(prompt, context)
#         with st.chat_message("assistant"):
#             st.markdown(response)
#         st.session_state.messages.append({"role": "assistant", "content": response})
#
#     # --- Clear Button ---
#     if st.button("Clear Conversation"):
#         clear_conversation()
#
#     # --- User Preferences ---
#     with st.sidebar:
#         st.header("Preferences")
#         background_color = st.color_picker("Background Color", value=user_prefs["background_color"])
#         user_prefs["background_color"] = background_color
#         save_user_prefs(user_prefs)
#
# if __name__ == "__main__":
#     main()

