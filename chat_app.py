import streamlit as st
from datetime import datetime
import pyperclip  # for copy functionality
import json
from pathlib import Path
import openai
from anthropic import Anthropic

# Page config
st.set_page_config(
    page_title="AI Interview Chat",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://your-help-url',
        'Report a bug': "https://your-bug-report-url",
        'About': "Your custom about message"
    }
)

# Add custom CSS
st.markdown("""
    <style>
    .stTextInput > label {
        font-size: 18px;
        font-weight: bold;
        color: #0f52ba;
    }
    .stMarkdown {
        font-size: 16px;
    }
    .stButton > button {
        background-color: #0f52ba;
        color: white;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "interview_instructions" not in st.session_state:
    st.session_state.interview_instructions = ""
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    api_choice = st.selectbox("API Provider", ["openai", "anthropic"])
    api_key = st.text_input("API Key", type="password")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_length = st.slider("Max Response Length", 100, 750, 250, 
                          help="Maximum number of tokens in the response")
    
    # Interview instructions
    st.header("Interview Context")
    st.session_state.interview_instructions = st.text_area(
        "Additional Context",
        value=st.session_state.interview_instructions,
        height=100,
        help="Provide background information about the interviewee"
    )
    
    # Move Clear and Export buttons to sidebar bottom
    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.messages = []
    
    chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
    st.download_button(
        label="Export Chat",
        data=chat_text,
        file_name="interview_chat.txt",
        mime="text/plain"
    )

# Define your system prompt
SYSTEM_PROMPT = """
You are being interviewed by the user who is a professor from one of the world's leading universities specializing in qualitative research methods. Based on the additional context provided, you will take on the role of the specified interviewee.

The interview aims to explore the different dimensions and factors that influenced your (the interviewee's) choice of profession and career path.

During the interview:
- Respond as the specified persona, maintaining consistency with the background provided
- Do NOT break character
- Answer questions openly and honestly, sharing relevant experiences, events, people, places, or practices that influenced your decisions
- Provide specific details and examples to give a deeper understanding of your professional journey
- Feel free to express views and beliefs that align with your assigned persona
- Match the tone, style, slang,and personality of the persona
- If you find any question unclear, don't hesitate to ask for clarification
- Stay in character throughout the conversation
- If relevant to the question, add interesting professional anecdotes or examples, but keep it short and concise

The professor (user) will begin the interview with their questions. Do not break character, do not refer to the user as 'user' or Professor. Be polite and casual."""

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Before the chat input section
st.markdown("---")
col1, col2 = st.columns([1,1])
with col2:
    if "total_tokens" in st.session_state:
        st.markdown(f"Tokens Used: {st.session_state.total_tokens}")

# Before the chat input section, add this:
if not st.session_state.initialized and st.session_state.interview_instructions:
    # Add a hidden system message to guide initial responses
    st.session_state.messages.append({
        "role": "system",
        "content": f"{SYSTEM_PROMPT}\n\nAdditional Context:\n{st.session_state.interview_instructions}"
    })
    st.session_state.initialized = True

# Define the interview function before using it
def interview(api, system_prompt, temperature, api_key, messages):
    if api == "openai":
        # Initialize client without proxies
        client = openai.OpenAI(
            api_key=api_key,
        )
        messages_formatted = [{"role": "system", "content": system_prompt}]
        messages_formatted.extend([
            {"role": m["role"], "content": m["content"]} 
            for m in messages
        ])
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages_formatted,
            temperature=temperature,
            max_tokens=max_length,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    elif api == "anthropic":
        client = Anthropic(api_key=api_key)
        
        # Format messages for Claude
        messages_text = system_prompt + "\n\n"
        for m in messages:
            role = "Human" if m["role"] == "user" else "Assistant"
            messages_text += f"{role}: {m['content']}\n\n"
        
        with client.messages.stream(
            model="claude-3-sonnet-20240229",
            max_tokens=max_length,
            temperature=temperature,
            messages=[{"role": "user", "content": messages_text}]
        ) as stream:
            for chunk in stream:
                if chunk.delta.text:
                    yield chunk.delta.text

# Chat input
if prompt := st.chat_input("Your message"):
    if not api_key:
        st.error("Please enter an API key in the sidebar.")
    else:
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            try:
                # Combine system prompt with interview instructions
                full_system_prompt = SYSTEM_PROMPT
                if st.session_state.interview_instructions:
                    full_system_prompt = f"{SYSTEM_PROMPT}\n\nAdditional Context:\n{st.session_state.interview_instructions}"
                
                full_response = ""
                for response_chunk in interview(
                    api=api_choice,
                    system_prompt=full_system_prompt,
                    temperature=temperature,
                    api_key=api_key,
                    messages=st.session_state.messages
                ):
                    full_response += response_chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error: {str(e)}")