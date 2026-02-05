import streamlit as st
import os
import google.generativeai as genai
from openai import OpenAI
from pinecone import Pinecone

# --- 1. CONFIGURATION ---
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    pass

# Pinecone Details
INDEX_HOST = "veraibot1536-o0tqsfu.svc.aped-4627-b74a.pinecone.io"
NAMESPACES = ["book-mybook-cs", "blog-cs", "podcast_cs"]

# --- 2. SETUP PAGE ---
st.set_page_config(page_title="AI Kouč Věra (Gemini)", page_icon="🌱")
st.title("🌱 AI Kouč (Věra Svach)")

# --- 3. INITIALIZE CLIENTS ---
@st.cache_resource
def init_clients():
    if "GOOGLE_API_KEY" not in os.environ:
        st.error("Missing Google API Key in Secrets.")
        st.stop()
        
    # DIRECT CONFIGURATION (Matches your local script)
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    
    return OpenAI(), Pinecone(api_key=PINECONE_API_KEY)

client, pc = init_clients()
index = pc.Index(host=INDEX_HOST)

def get_embedding(text):
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=[text], model="text-embedding-3-small"
    )
    return response.data[0].embedding

def retrieve_context(query):
    try:
        query_vector = get_embedding(query)
    except Exception:
        return "", ""

    all_matches = []
    for ns in NAMESPACES:
        try:
            results = index.query(
                namespace=ns, vector=query_vector, top_k=3, include_metadata=True
            )
            for match in results['matches']:
                match['source_namespace'] = ns
                all_matches.append(match)
        except Exception:
            pass

    sorted_matches = sorted(all_matches, key=lambda x: x['score'], reverse=True)
    contexts = []
    debug_text = ""
    for match in sorted_matches[:5]:
        text_content = match['metadata'].get('text', '')
        source = match.get('source_namespace', 'unknown')
        if text_content:
            contexts.append(text_content)
            debug_text += f"--- Source: {source} ---\n{text_content[:200]}...\n\n"
    return "\n\n".join(contexts), debug_text

def get_gemini_response(user_input, chat_history):
    # 1. Retrieve Context
    context, debug_text = retrieve_context(user_input)
    if not context: context = "V databázi nebyla nalezena přímá odpověď."

    # 2. FORCE THE WORKING MODEL
    # We are not asking Google "what is available". We are telling it "Use Flash".
    # This matches your working local script.
    model = genai.GenerativeModel('gemini-1.5-flash')

    # 3. Prepare History
    gemini_history = []
    for msg in chat_history[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=gemini_history)

    # 4. Prompt
    system_instruction = "Jsi AI kouč založený na filozofii Věry Svach. Buď empatická, stručná a mluv česky."
    final_prompt = f"{system_instruction}\n\nKONTEXT:\n{context}\n\nOTÁZKA:\n{user_input}"

    try:
        response = chat.send_message(final_prompt)
        return response.text, debug_text
    except Exception as e:
        return f"Chyba Gemini: {str(e)}", ""

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Napište svou otázku..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Gemini přemýšlí..."):
            response_text, sources = get_gemini_response(prompt, st.session_state.messages)
            st.markdown(response_text)
            
    st.session_state.messages.append({"role": "assistant", "content": response_text})
