import streamlit as st
import os
import google.generativeai as genai
from openai import OpenAI
from pinecone import Pinecone

# --- 1. CONFIGURATION ---
try:
    # We still need OpenAI for the Embeddings (Search)
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    pass

# Configure Google Gemini
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

INDEX_HOST = "veraibot1536-o0tqsfu.svc.aped-4627-b74a.pinecone.io"
NAMESPACES = ["book-mybook-cs", "blog-cs", "podcast_cs"]

# --- 2. SETUP PAGE ---
st.set_page_config(page_title="AI Kouč Věra (Gemini)", page_icon="🌱")
st.title("🌱 AI Kouč (Powered by Google Gemini)")
st.markdown("Zeptejte se na cokoliv ohledně seberozvoje, stresu nebo mindfulness.")

# --- 3. INITIALIZE CLIENTS ---
@st.cache_resource
def init_clients():
    if "OPENAI_API_KEY" not in os.environ:
        st.error("Chybí OpenAI API klíč (pro vyhledávání).")
        st.stop()
    return OpenAI(), Pinecone(api_key=PINECONE_API_KEY)

client, pc = init_clients()
index = pc.Index(host=INDEX_HOST)

def get_embedding(text):
    # We MUST use OpenAI embeddings because the database was built with them
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

def get_gemini_response(user_input, chat_history):
    # 1. Get Context
    context, debug_text = retrieve_context(user_input)
    
    if not context:
        context_message = "V databázi jsem nenašla přímou odpověď."
    else:
        context_message = context

    # 2. Configure the Gemini Model
    # 'gemini-1.5-flash' is fast and cheap. You can also use 'gemini-1.5-pro' for better reasoning.
    model = genai.GenerativeModel('gemini-pro')

    # 3. System Prompt (Gemini Instructions)
    system_instruction = f"""
    Jsi AI kouč založený na filozofii Věry Svach.
    
    KONTEXT Z DATABÁZE (pro aktuální otázku):
    {context_message}
    
    INSTRUKCE:
    1. Primárně vycházej z kontextu výše.
    2. Pokud uživatel navazuje na předchozí konverzaci, použij historii chatu.
    3. Buď empatická, stručná a mluv česky.
    """

    # 4. Convert Chat History to Gemini Format
    # Gemini uses specific roles: "user" and "model" (instead of "assistant")
    gemini_history = []
    
    # Add system instruction essentially as the first context setting
    # (Gemini often handles system instructions via 'system_instruction' param in newer versions, 
    # but injecting it into the first prompt is a robust fallback)
    
    for msg in chat_history[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})
        
    # Start a chat session with the history
    chat = model.start_chat(history=gemini_history)
    
    # 5. Send the new message (User Query + System/Context Injection)
    final_prompt = f"{system_instruction}\n\nOTÁZKA UŽIVATELE:\n{user_input}"
    
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
        if "sources" in message:
             with st.expander("🔍 Zobrazit zdroje (Historie)"):
                st.text(message["sources"])

if prompt := st.chat_input("Napište svou otázku..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Hledám v databázi a přemýšlím (Gemini)..."):
            response_text, sources_text = get_gemini_response(prompt, st.session_state.messages)
            st.markdown(response_text)
            
            with st.expander("🔍 Zobrazit použité texty (Důkaz)"):
                st.text(sources_text)
            
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "sources": sources_text
    })
