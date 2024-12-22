#!/usr/bin/env python
# coding: utf-8

# In[20]:


import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sympy import sympify, solve
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import logging


# In[21]:


ssl._create_default_https_context = ssl._create_unverified_context
logging.getLogger('streamlit').setLevel(logging.ERROR)


# In[22]:


nltk.download('punkt')


# In[23]:


vectorizer_path = "vectorizer.joblib"
model_path = "model.joblib"


# In[24]:


file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)


# In[25]:


if os.path.exists(vectorizer_path) and os.path.exists(model_path):
    # Load the saved vectorizer and model
    vectorizer = joblib.load(vectorizer_path)
    clf = joblib.load(model_path)
else:
    # Create the vectorizer and classifier
    vectorizer = TfidfVectorizer()
    clf = LogisticRegression(random_state=0, max_iter=10000)

    # Preprocess the data
    tags = []
    patterns = []
    for intent in intents:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(intent['tag'])

    x_train, x_test, y_train, y_test = train_test_split(patterns, tags, test_size=0.2, random_state=42)

    x_train_vectorized = vectorizer.fit_transform(x_train)
    clf.fit(x_train_vectorized, y_train)

    # Save the trained model and vectorizer
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(clf, model_path)


# In[26]:


def math_solver(query):
    try:
        # Check if the query is a basic arithmetic expression
        if re.match(r'^[0-9+\-*/().\s]+$', query):
            # Evaluate the arithmetic expression
            result = eval(query)
            return f"The result is: {result}"
        else:
            # Treat as a symbolic equation and solve it
            expression = sympify(query)
            solution = solve(expression)
            return f"The solution is: {solution}"
    except Exception as e:
        return f"Sorry, I couldn't solve that. Error: {e}"


# In[27]:


def handle_multiple_intents(user_input):
    responses = []
    input_vector = vectorizer.transform([user_input])
    predicted_tags = clf.predict(input_vector)

    for tag in predicted_tags:
        for intent in intents:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                responses.append(response)
                break  # Break after finding the first matching intent

    return responses if responses else ["I'm not sure how to respond to that."]


# In[28]:


def chatbot(user_input):
    try:
        # Check for math-related queries
        if "math" in user_input.lower() or re.search(r'\d', user_input):
            return math_solver(user_input)
        
        responses = handle_multiple_intents(user_input)
        return " ".join(responses)  # Combine responses into a single string
    except Exception as e:
        return f"Sorry, I couldn't process that. Error: {e}"


# In[30]:


def main():
    # Custom CSS for background and text color
    st.markdown("""
        <style>
        body {
            background-color: #d3eba7;  /* Change this color code to any color you want */
            color: #1f034d; /* Dark gray text */
        }
        .sidebar .sidebar-content {
            background-color: #95b310; /* Sidebar background color */
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("Rosh! 😊 Chatbot with NLP and Math Solver")
    
    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    # Home Page
    if choice == "Home":
        st.write("Welcome to the chatbot,How can is assist you? Please type a message and enter to start the conversation.")
        
        # Check if the chat_log.csv file exists, and create it if not
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])
        
        user_input = st.text_input("You:")
        
        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None)

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])
        
            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Page
    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history found.")

    # About Page
    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to a conversation, including solving mathematical problems.")
        st.subheader("Features:")
        st.write("""
        1. NLP techniques and Logistic Regression are used to train the chatbot.
        2. Streamlit framework is used to build the web interface.
        3. Integrated math solver for handling mathematical queries.
        """)
        
if __name__ == '__main__':
    main()

