import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from sqlalchemy import create_engine, text
import sqlite3
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
import pandas as pd
from langchain.callbacks import StreamlitCallbackHandler

# Load environment variables
load_dotenv()

# Setting up the page configuration with title and icon
st.set_page_config(page_title="ML Project", page_icon="✨")

# Custom CSS for modern design
st.markdown("""
    <style>

        .header {
            background-color: #1f1f1f;
            color: #e0e0e0;
            padding: 10px;
            text-align: center;
            border-radius: 10px;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .stDataFrame {
            background-color: #2e2e2e;
            color: #d0d0d0;
        }
        .stDataFrame thead th {
            background-color: #4a4a4a;
            color: #ffffff;
        }
        .stDataFrame tbody tr:nth-child(even) {
            background-color: #3c3c3c;
        }
        .stDataFrame tbody tr:nth-child(odd) {
            background-color: #2e2e2e;
        }
        .stChat {
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .user {
            background-color: grey;
            color: #ffffff;
            text-align: right;
        }
        .assistant {
            background-color: lightgrey;

            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# Setting up the title of the app
# st.markdown('<div class="header"><h1>Chat-Mate: Conversational Analytics Chatbot📝</h1></div>', unsafe_allow_html=True)

st.sidebar.title("Settings⚙️")

# Setting up the title of the app
st.markdown('<h1 class="header-title">Chat-Mate...Conversational Analytics Chatbot📝</h1>', unsafe_allow_html=True)# Retrieve the Groq API key from the .env file
api_key = os.getenv("GROQ_API_KEY")

# Check if the API key is provided
if not api_key:
    st.error("Please set the Groq API key in the .env file.")
    st.stop()

# Initialize the Groq LLM
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

# Function to configure SQLite database
@st.cache_resource(ttl="2h")
def configure_db():
    dbfilepath = (Path(__file__).parent / "analytics_db").absolute()
    creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
    return SQLDatabase(create_engine("sqlite:///", creator=creator))

# Configure DB
db = configure_db()

# Function to list all tables in the database
def list_tables(db):
    engine = db._engine
    with engine.connect() as conn:
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
        tables = result.fetchall()
        return [table[0] for table in tables]

# List and display tables in the database
tables = list_tables(db)
st.subheader("Available Tables in the Database:")
st.write(tables)

# Displaying the database content in a tabular format
def display_database_table(db, table_name):
    query = f"SELECT * FROM {table_name} LIMIT 50"
    engine = db._engine  # Access the SQLAlchemy engine
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    st.subheader(f"Data from {table_name}:")
    st.dataframe(df.style.set_table_styles([{
        'selector': 'thead th',
        'props': 'background-color: #4a4a4a; color: #ffffff;'
    }, {
        'selector': 'tbody tr:nth-of-type(odd)',
        'props': 'background-color: #2e2e2e;'
    }, {
        'selector': 'tbody tr:nth-of-type(even)',
        'props': 'background-color: #3c3c3c;'
    }, {
        'selector': 'td',
        'props': 'color: #d0d0d0;'
    }]), use_container_width=True)

# Replace 'your_table_name' with an actual table name from the list
if tables:
    display_database_table(db, table_name=tables[0])  # Display the first table as an example
else:
    st.warning("No tables found in the database.", icon="🚫")

# SQL toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,  # Added to handle parsing errors
    max_iterations=10,           # Increase this value if needed
    timeout=300
)

# Session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "Chat-Mate", "content": "How can I help you?"}]

# Display chat history messages
for msg in st.session_state["messages"]:
    st.markdown(f'<div class="stChat {msg["role"]}">{msg["content"]}</div>', unsafe_allow_html=True)

# Input for user query
user_query = st.chat_input(placeholder="Ask anything from the database")

# If user query is submitted
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.markdown(f'<div class="stChat user">{user_query}</div>', unsafe_allow_html=True)

    # Generate response from agent
    with st.chat_message("Chat-Mate"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        try:
            response = agent.run(user_query, callbacks=[streamlit_callback])
            st.session_state.messages.append({"role": "Chat-Mate", "content": response})

            # Ensure the response is in tabular format
            if isinstance(response, list):
                if all(isinstance(i, tuple) for i in response) and len(response) > 0:
                    # Assuming the first tuple contains the headers
                    headers = [f"Column {i+1}" for i in range(len(response[0]))]
                    df = pd.DataFrame(response, columns=headers)
                    st.dataframe(df.style.set_properties(**{'color': 'white', 'background-color': 'black'}))
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


# Container for buttons at the bottom of the page
with st.container():
    # Adding some spacing before the buttons
    st.write("<br>", unsafe_allow_html=True)
    
    # Clear Chat History Button
    if st.sidebar.button("Clear Chat History"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you sir?"}]
    
    # Download Chat History Button
    chat_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state["messages"]])
    st.sidebar.download_button(
        label="Download Chat History",
        data=chat_history,
        file_name="chat_history.txt",
        mime="text/plain"
    )

