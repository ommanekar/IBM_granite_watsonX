import streamlit as st
import time
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import tempfile
import docx
import PyPDF2
import re
import uuid

# IBM Watson Credentials
IBM_API_KEY = "8sNomPm68AOIXhU84fABUnE5OioYvQHidhSt8iJZV8qA"
IBM_PROJECT_ID = "65e64ad2-cbdd-4e5e-87c4-0b05eb1fbd13"
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"

credentials = Credentials(url=WATSONX_URL, api_key=IBM_API_KEY)
model_id = "ibm/granite-3-3-8b-instruct"
parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 350,
    "min_new_tokens": 0,
    "temperature": 0.45,
    "top_k": 50,
    "top_p": 0.88,
    "repetition_penalty": 1
}
model = ModelInference(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=IBM_PROJECT_ID
)

def extract_text(uploaded_file):
    filetype = uploaded_file.name.split('.')[-1].lower()
    if filetype == "txt":
        return uploaded_file.read().decode(errors="ignore")
    elif filetype == "pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    elif filetype == "docx":
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        doc = docx.Document(tmp_path)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return ""

def granite_instruct_response(prompt: str) -> str:
    try:
        response = model.generate_text(prompt=prompt, guardrails=True)
        if isinstance(response, dict):
            return response.get('results', [{}])[0].get('generated_text', '').strip()
        elif isinstance(response, list):
            return response[0].get('generated_text', '').strip()
        return str(response)
    except Exception as e:
        return f"IBM Granite API error: {e}"

def summarize_document(text):
    prompt = (
        "Summarize the following business document as 5-8 professional bullet points. "
        "Do NOT include any section headers, introductions, the word 'SUMMARY', or any bullet with fewer than three words. "
        "Each bullet must be a complete, informative statement (no single words). "
        "Respond with markdown-formatted bullets ONLY, one bullet per line. Document:\n"
        + text[:6000]
    )
    return granite_instruct_response(prompt)

def parse_markdown_bullets(text):
    bullets = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("-", "*", "•")):
            content = line[1:].strip()
            if (
                len(content.split()) < 3 or
                content.lower().startswith(("summary", "document", "conclusion", "action", "item"))
            ):
                continue
            bullets.append(f"- {content}")
    if not bullets:
        sentences = [ln.strip() for ln in text.splitlines() if len(ln.split()) > 3]
        bullets = [f"- {s}" for s in sentences]
    return "\n".join(bullets)

def generate_ai_response(user_prompt, doc_summary, thinking_mode):
    if not doc_summary:
        return "Please upload and summarize a document first."

    prompt_templates = {
        "Creative": (
            "Based only on the following summary, generate 2–3 imaginative yet practical business ideas "
            "to address the user's prompt. After each idea, briefly explain why it is novel and valuable in this business context.\n\n"
            "SUMMARY:\n{summary}\nPROMPT: {prompt}"
        ),
        "Diverse": (
            "Using only the summary, suggest 2–3 unconventional or unique business ideas based on the user's prompt. "
            "After each idea, explain in 1-2 lines what makes it stand out in context.\n\n"
            "SUMMARY:\n{summary}\nPROMPT: {prompt}"
        ),
        "Lateral": (
            "Apply lateral thinking: based on this summary, generate 2–3 surprising, non-obvious solutions to the prompt (no repeats). "
            "Give a 1-line explanation of each idea's value for the business.\n\n"
            "SUMMARY:\n{summary}\nPROMPT: {prompt}"
        )
    }

    if thinking_mode not in prompt_templates:
        return "Invalid thinking mode selected."

    full_prompt = prompt_templates[thinking_mode].format(summary=doc_summary, prompt=user_prompt)
    return granite_instruct_response(full_prompt)

# Streamlit UI Setup
st.set_page_config(page_title="AI Brainstorming Buddy", layout="wide")
if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session" not in st.session_state:
    session_id = str(uuid.uuid4())
    st.session_state.current_session = session_id
    st.session_state.sessions[session_id] = {
        "name": "Session 1",
        "messages": [],
        "doc_summary": ""
    }

def get_current_session():
    return st.session_state.sessions[st.session_state.current_session]

def handle_user_input(thinking_mode=None):
    user_input = st.session_state.msg_input.strip()
    if not user_input:
        return
    session = get_current_session()
    session["messages"].append({"role": "user", "content": user_input})
    ai_response = generate_ai_response(user_input, session["doc_summary"], thinking_mode)
    session["messages"].append({"role": "bot", "content": f"**{thinking_mode} Thinking**\n\n{ai_response}"})
    st.session_state.msg_input = ""

# Sidebar
with st.sidebar:
    st.title("📂 Upload Document")
    uploaded_file = st.file_uploader("PDF, Word, or TXT only", type=['pdf', 'docx', 'txt'])
    if uploaded_file:
        with st.spinner("Extracting and summarizing..."):
            file_text = extract_text(uploaded_file)
            summary = summarize_document(file_text)
            get_current_session()["doc_summary"] = parse_markdown_bullets(summary)
        st.success("Document summarized ✔️")

    st.markdown("---")
    st.subheader("💬 Chat Sessions")
    for sid, sess in st.session_state.sessions.items():
        if st.button(sess["name"], key=sid):
            st.session_state.current_session = sid

    if st.button("➕ New Session"):
        new_id = str(uuid.uuid4())
        new_name = f"Session {len(st.session_state.sessions)+1}"
        st.session_state.sessions[new_id] = {
            "name": new_name,
            "messages": [],
            "doc_summary": ""
        }
        st.session_state.current_session = new_id
        st.experimental_rerun()

    if st.button("🗑️ Reset Current"):
        current = get_current_session()
        current["messages"] = []
        current["doc_summary"] = ""
        st.experimental_rerun()

# Header
st.markdown("<h2 style='margin-bottom:0;'>AI Brainstorming Buddy</h2>", unsafe_allow_html=True)
st.caption("Inject creative, diverse, or lateral ideas into your team using real business documents.")

# Summary
if get_current_session()["doc_summary"]:
    with st.expander("📄 SUMMARY:", expanded=True):
        st.markdown(get_current_session()["doc_summary"])

# Thinking Mode Buttons
col1, col2, col3 = st.columns([1, 1, 1])
if col1.button("🎨 Creative"):
    handle_user_input("Creative")
if col2.button("🌐 Diverse"):
    handle_user_input("Diverse")
if col3.button("🔀 Lateral"):
    handle_user_input("Lateral")

# Chat Input
col1, col2 = st.columns([6, 1])
col1.text_input("Your Message", key="msg_input")
col2.button("Send", on_click=lambda: handle_user_input("Creative"))  # Defaulting to Creative

# Chat Display
USER_ICON = "🧑"
BOT_ICON = "🤖"
for msg in get_current_session()["messages"]:
    role = msg['role']
    icon = USER_ICON if role == "user" else BOT_ICON
    bubble_class = "user-bubble" if role == "user" else "bot-bubble"
    row_justify = "flex-end" if role == "user" else "flex-start"
    st.markdown(
        f"<div class='msg-row' style='justify-content:{row_justify};'>"
        f"<div class='chat-bubble {bubble_class}'>{msg['content']}</div></div>",
        unsafe_allow_html=True
    )

# Styles
st.markdown("""<style>
    .chat-bubble {border-radius:15px; padding:10px 16px; margin:6px; max-width:70%; font-size:1.07rem;}
    .user-bubble {background:#364F6B; color:#fff; align-self: flex-end;}
    .bot-bubble {background:#F4F6FB; color:#222; align-self: flex-start;}
    .msg-row {display:flex; flex-direction:row;}
</style>""", unsafe_allow_html=True)
