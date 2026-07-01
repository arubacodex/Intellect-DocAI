import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Intellect-Doc AI",
    page_icon="🧠",
    layout="wide", 
)

# 🎨 ENGINE CSS MODIFICATIONS FOR PERFECT WIDTH AND SPACING
st.markdown("""
    <style>
    /* Strict global grid layout centering matching your app limits */
    .stMainBlockContainer {
        max-width: 940px !important; /* Unified width boundary for the entire grid layout */
        padding-top: 2rem !important;
        margin: 0 auto !important;
    }

    /* Shifting Mesh Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #060202 0%, #110303 40%, #020612 100%) !important;
        color: #ffffff !important;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    /* Document Global Text Output Protection */
    .stMarkdown, p, span, label, li, small, h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Branding Header Setup */
    .app-header {
        text-align: center;
        margin-bottom: 1rem;
    }
    .app-title {
        color: #ff2b2b !important;
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        text-shadow: 0 0 20px rgba(255, 43, 43, 0.4);
        margin: 0 0 6px 0;
    }
    .app-subtitle {
        color: #b0b0b0 !important;
        font-size: 0.95rem !important;
        margin: 0;
    }
    
    /* Push controls lower down away from title and description text */
    [data-testid="stHorizontalBlock"] {
        align-items: center !important;
        gap: 1rem !important;
        margin-top: 2.5rem !important;
        margin-bottom: 1.5rem !important;
    }

    /* File uploader container with high-contrast background and visible text */
    div[data-testid="stFileUploader"] {
        background-color: #ffffff !important; 
        border: 2px solid #ff2b2b !important;
        border-radius: 8px !important;
        padding: 0px 14px !important;
        height: 52px !important;
        display: flex !important;
        align-items: center !important;
        box-shadow: 0 0 15px rgba(255, 43, 43, 0.25) !important;
    }
    
    /* Strip native inner padding from the file block */
    div[data-testid="stFileUploader"] > section {
        padding: 0 !important;
        min-height: auto !important;
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
    }
    
    /* Forces all file upload typography and prompts to display as crisp black */
    div[data-testid="stFileUploader"] *, 
    div[data-testid="file-uploader-dropzone"] *,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] div,
    div[data-testid="stFileUploader"] p {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    
    /* File browse button adjustment */
    div[data-testid="stFileUploader"] button {
        background-color: rgba(0, 0, 0, 0.1) !important;
        color: #000000 !important;
        border: 1px solid rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Knowledge base button neon hover animation properties */
    div.stButton > button {
        background: linear-gradient(90deg, #990000 0%, #ff2b2b 100%) !important;
        color: #ffffff !important;
        border: 1px solid #ff4d4d !important;
        border-radius: 8px !important;
        height: 52px !important; 
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        width: 100%;
        margin: 0 !important; 
        box-shadow: 0 0 15px rgba(255, 43, 43, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #cc0000 0%, #ff1a1a 100%) !important;
        box-shadow: 0 0 30px #ff2b2b, 0 0 50px rgba(255, 43, 43, 0.5) !important;
        transform: scale(1.01);
    }
    
    /* Eliminates the native white background bar behind the bottom input */
    div[data-testid="stBottom"] {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* 🔥 FORCES THE FIXED BOTTOM CONTAINER TO STAY RESTRAINED, CENTERED, AND EQUAL WIDTH */
    div[data-testid="stBottom"] > div {
        max-width: 940px !important; /* Forces layout match with upper components */
        margin: 0 auto !important; /* Perfect Center Alignment Alignment */
        padding-left: 0 !important;
        padding-right: 0 !important;
        background-color: transparent !important;
        background: transparent !important;
    }
    
    /* 🔥 CHAT BOX BOX ALIGNMENT SYMMETRY FIXES */
    div[data-testid="stChatInput"] {
        border: 2px solid rgba(255, 43, 43, 0.6) !important;
        border-radius: 8px !important;
        background-color: #ffffff !important; 
        box-shadow: 0 4px 25px rgba(0, 0, 0, 0.8) !important;
        padding: 4px !important;
        width: 60% !important; /* Fills outer matching center frame container */
        margin: 0 auto !important;
    }
    div[data-testid="stChatInput"] textarea {
        color: #000000 !important; 
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        background-color: transparent !important;
    }
    
    /* Chat Conversation Bubble Rows */
    div[data-testid="stChatMessage"] {
        background-color: #0b0b0d !important;
        border: 1px solid #1a1a1e !important;
        border-radius: 10px !important;
        padding: 14px !important;
        margin-bottom: 0.8rem !important;
    }
    div[data-testid="stChatMessage"]([data-is-assistant="true"]) {
        border-left: 3px solid #ff2b2b !important;
        background: linear-gradient(90deg, #140808 0%, #0b0b0d 100%) !important;
    }

    /* Core Section Label Layouts */
    .chat-section-title {
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. BRANDING BANNER
st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🧠 Intellect-Doc AI</h1>
        <p class="app-subtitle">Instant Document Intelligence RAG Pipeline powered by Groq & LangChain</p>
    </div>
""", unsafe_allow_html=True)

# 3. EXTRACTION MATRIX FOR API SECURITY
groq_api_key = None
try:
    if "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not groq_api_key:
    groq_api_key = st.sidebar.text_input("🔑 System Authentication Access Key", type="password")
    if not groq_api_key:
        st.info("⚡ Configuration Needed: Please set your GROQ_API_KEY credential attributes to proceed.")
        st.stop()

# 4. INITIALIZE RECOVERY SYSTEM STATE WORKSPACE
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. INITIALIZE CORE MODEL INFRASTRUCTURE RESOURCES
@st.cache_resource
def load_llm_and_embeddings(api_key):
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.1
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return llm, embeddings

try:
    llm, embeddings = load_llm_and_embeddings(groq_api_key)
except Exception as e:
    st.error(f"Ecosystem Setup Error: {e}")
    st.stop()

prompt = ChatPromptTemplate.from_template(
    """
    You are Intellect Doc AI.
    Answer ONLY using the provided context. If the answer is missing, state precisely that you can't find it.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
)

# 6. VECTOR COMPILATION LOADER
def build_rag(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name
        try:
            loader = PyPDFLoader(temp_path)
            documents.extend(loader.load())
        except Exception:
            pass
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    if not documents:
        return False

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, embeddings)
    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return True

# 7. VISUAL SYSTEM ROW (Upload and Build Button Row)
layout_col1, layout_col2 = st.columns([6, 4])

with layout_col1:
    uploaded_files = st.file_uploader(
        "Upload Source PDF Documentation Records",
        type=["pdf"],
        accept_multiple_files=True,
        key="main_uploader",
        label_visibility="collapsed"
    )

with layout_col2:
    build_pressed = st.button("🚀 Build Knowledge Base")

if build_pressed:
    if not uploaded_files:
        st.warning("Select target assets prior to compilation.")
    else:
        with st.spinner("Processing framework vectors..."):
            success = build_rag(uploaded_files)
            if success:
                st.success("🤖 Knowledge base built.")

# 8. ACTIVE MONITOR DISPLAY FEED
st.markdown("<div class='chat-section-title'>✨Execution Engine</div>", unsafe_allow_html=True)

chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

question = st.chat_input("Query context records...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(question)
            
        if st.session_state.retriever is None:
            answer = "⚠️ System Notification: Mount active documentation nodes inside the unified row interface above."
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                with st.spinner("🧠 Scanning active nodes..."):
                    try:
                        docs = st.session_state.retriever.invoke(question)
                        context = "\n\n".join(doc.page_content for doc in docs)
                        
                        chain = prompt | llm | StrOutputParser()
                        answer = chain.invoke({"context": context, "question": question})
                        
                        response_placeholder.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        response_placeholder.markdown(f"❌ Processing Fault Encountered: {e}")
