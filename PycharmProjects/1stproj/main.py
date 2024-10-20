# from dotenv import load_dotenv
# import os
# import google.generativeai as genai
#
# # Load the environment variables from .env file
# load_dotenv()
#
# # Configure the API key
# api_key = os.getenv("api_key")
# if not api_key:
#     raise ValueError("API key not found. Make sure it's set in the .env file.")
#
# genai.configure(api_key=api_key)
# model = genai.GenerativeModel('gemini-1.5-flash')
#
# # Initialize a list to store the conversation history
# conversation_history = []
#
# # Loop to interact with the AI model
# while True:
#     user_input = input("User: ")
#     if user_input.lower() == 'exit':
#         break
#
#     # Append the user's input to the conversation history list
#     conversation_history.append(f"User: {user_input}")
#
#     # Combine the conversation history into a single string for the model
#     history_string = "\n".join(conversation_history)
#
#     # Generate a response using the model with the full conversation history
#     response = model.generate_content(history_string)
#
#     # Assuming the response contains the 'text' attribute
#     ai_response = response.text if hasattr(response, 'text') else "Error: Response format unexpected."
#
#     # Append the AI's response to the conversation history list
#     conversation_history.append(f"AI: {ai_response}")
#
#     # Print the AI's response
#     print("AI:", ai_response)

# from io import BytesIO  #to conversion into bites(it is for transmission over network
# import streamlit as st
# from PIL import Image #to play with images
# from rembg import remove #to remove background
#
# st.set_page_config(layout="wide", page_title="Image Background Remover")
#
# st.title(" Image Background Remover")
#
# st.sidebar.write("## Upload and download :gear:")
#
# # Create the columns
# col1, col2 = st.columns(2)
#
# # Download the fixed image
# def convert_image(img):
#     buf = BytesIO()
#     img.save(buf, format="PNG")
#     byte_im = buf.getvalue()
#     return byte_im
#
# # Package the transform into a function
# def fix_image(upload):
#     image = Image.open(upload)
#     col1.write("Your Image")
#     col1.image(image)
#
#     fixed = remove(image)
#     col2.write("New Image")
#     col2.image(fixed)
#     st.sidebar.download_button(
#         "Download new image", convert_image(fixed), "fixed.png", "image/png"
#     )
#
# # Create the file uploader
# my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
#
# # # Fix the image!
# if my_upload is not None:
#     fix_image(upload=my_upload)
#
# from dotenv import load_dotenv
# import os
# import google.generativeai as genai
# import streamlit as st
#
# load_dotenv()
#
# # Configure the API key
# api_key = os.getenv("api_key")
# if not api_key:
#     raise ValueError("API key not found. Make sure it's set in the .env file.")
#
# genai.configure(api_key=api_key)
# model = genai.GenerativeModel('gemini-1.5-flash')
#
# st.title("Text Summarizer")
# txt = txt_input = st.text_area('Enter your text', '', height=200)
# st.write(f"You wrote {len(txt)} characters.")
# if st.button("Submit to Summarize"):
#     prompt = "You will act as an text summarizer. I will provide you text, you will generate its summary in 5 lines."
#     ai_input = prompt + txt
#     response = model.generate_content(ai_input)
#     st.markdown(response.text)
#
# ---------------------------------------------------------------------
