import streamlit as st
import os
import google.generativeai as genai
from openai import OpenAI
from pinecone import Pinecone

# --- 1. CONFIGURATION ---
# We load keys from Streamlit Secrets
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    # If running locally without secrets.toml, you might want to hardcode keys here for testing
    # But for deployment, Secrets are best.
    pass

# Configure Pinecone
INDEX_HOST = "veraibot1536-o0tqsfu.svc.aped-4627-b74a.pinecone.io"
NAMESPACES = ["book-mybook-cs", "blog-cs", "podcast_cs"]

# --- 2. SETUP PAGE ---
st.set_page_config(page_title="AI Kouč Věra (Gemini)", page_icon="🌱")
st.title("🌱 AI Kouč (Věra Svach)")
st.markdown("Zeptejte se na cokoliv ohledně seberozvoje, stresu nebo mindfulness.")

# --- 3. INITIALIZE CLIENTS ---
@st.cache_resource
def init_clients():
    if "OPENAI_API_KEY" not in os.environ:
        st.error("Chybí API klíče. Zkontrolujte nastavení Secrets.")
        st.stop()
    
    # Configure Google
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    
    # Return OpenAI (for embeddings) and Pinecone
    return OpenAI(), Pinecone(api_key=PINECONE_API_KEY)

client, pc = init_clients()
index = pc.Index(host=INDEX_HOST)

# --- 4. ROBUST MODEL SELECTOR ---
def get_gemini_model():
    """
    Automatically finds a working model (Flash or Pro) to avoid 404 errors.
    """
    try:
        # Ask Google which models are available for this key
        available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Preferred order: Flash -> 1.5 Pro -> Old Pro
        preferences = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
        
        for pref in preferences:
            if pref in available:
                # Remove 'models/' prefix for initialization
                clean_name = pref.replace("models/", "")
                return genai.GenerativeModel(clean_name)
        
        # Fallback: Just try 'gemini-pro' if list lookup fails logic
        return genai.GenerativeModel('gemini-pro')
        
    except Exception:
        # Ultimate fallback
        return genai.GenerativeModel('gemini-pro')

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
        score = match.get('score', 0)
        
        if text_content:
            contexts.append(text_content)
            debug_text += f"--- [Zdroj: {source} | Relevence: {score:.2f}] ---\n{text_content[:300]}...\n\n"
            
    return "\n\n".join(contexts), debug_text

def get_response(user_input, chat_history):
    # 1. Retrieve Context
    context, debug_text = retrieve_context(user_input)
    
    if not context:
        context_message = "V databázi jsem nenašla přímou odpověď."
    else:
        context_message = context

    # 2. Prepare History for Gemini
    # Gemini uses 'user' and 'model' roles. Streamlit uses 'user' and 'assistant'.
    gemini_history = []
    for msg in chat_history[:-1]: # Skip the very last message (current prompt) to avoid duplication
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    # 3. Setup Model & Chat
    model = get_gemini_model()
    chat = model.start_chat(history=gemini_history)
    
    # 4. Create the System Prompt + Query
    # We inject the Persona and Context into the message itself
    system_instruction = """
    Jsi AI kouč založený na filozofii Věry Svach.
    Buď empatická, stručná a mluv česky.
    """
    
    final_prompt = f"""
    {system_instruction}
    
    KONTEXT Z DATABÁZE (pro aktuální otázku):
    {context_message}
    
    OTÁZKA UŽIVATELE:
    {user_input}
    
    Odpověz primárně na základě kontextu výše. Pokud uživatel navazuje na předchozí konverzaci (např. 'ona', 'to'), použij historii chatu.
    """
    
    try:
        response = chat.send_message(final_prompt)
        return response.text, debug_text
    except Exception as e:
        return f"Chyba Gemini: {str(e)}", ""

# --- 5. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
             with st.expander("🔍 Zobrazit zdroje (Historie)"):
                st.text(message["sources"])

# User Input
if prompt := st.chat_input("Napište svou otázku..."):
    # 1. Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("Hledám v databázi a přemýšlím (Gemini)..."):
            response_text, sources_text = get_response(prompt, st.session_state.messages)
            st.markdown(response_text)
            
            with st.expander("🔍 Zobrazit použité texty (Důkaz)"):
                st.text(sources_text)
            
    # 3. Save Assistant Message
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "sources": sources_text
    })
