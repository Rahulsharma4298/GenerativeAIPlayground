import base64

import streamlit as st

from chat_factory import chat
from rag import RAG

if 'retriever' not in st.session_state:
    st.session_state.retriever = None

def render_menu():
    type_ = st.session_state.get('selected_tab', 'chat')
    selected_model = st.session_state.get('selected_model', 'gemini-1.5.flash')
    temperature = st.session_state.get('temperature', 0.0)
    st.title(f":material/network_intelligence: :blue[RoyalAssist] - :red[{type_.title().replace('_', '')}]")
    # st.text(f"Model: {selected_model}")
    if type_ == 'rag':
        file = st.file_uploader('Upload File', type=('pdf',))
        if file:
            file_hash = hash(file.getvalue())
            if not st.session_state or st.session_state.get("file_hash") != file_hash:
                with st.spinner('Embedding ..'):
                    st.session_state.retriever = RAG.embed_docs(file.getvalue())
                    st.session_state.file_hash = file_hash
    image_file = st.file_uploader("Upload image", type=('jpg', 'png', 'jpeg'))
    encoded_image = None
    if user_input := st.chat_input():
        if image_file:
            st.image(image_file)
            encoded_image = base64.b64encode(image_file.read()).decode()
        st.chat_message('user').write(user_input)
        with st.spinner(':brain: Processing ..'):
            # resp = asyncio.run(chat(user_input,
            #             model=selected_model,
            #             type=type_,
            #             temperature=temperature,
            #             retriever=st.session_state.retriever,
            #             encoded_image=encoded_image))
            resp = chat(user_input,
                        model=selected_model,
                        type=type_,
                        temperature=temperature,
                        retriever=st.session_state.retriever,
                        encoded_image=encoded_image)
            if type_ == 'agent' and resp.startswith('https://www.youtube.com'):
                st.chat_message('assistant').video(resp)
            else:
                st.chat_message('assistant').write(resp)


st.set_page_config(page_title='RoyalAssist - Your AI Assistant')
with st.sidebar:
    st.header("📌 Menu", divider='red')

    tabs = {
        "Chat": "chat",
        "RAG": "rag",
        "Agent": "agent",
        "MCP-Agent": "mcp_agent",
        "ResearchAssistant": "research_assistant"
    }

    for label, value in tabs.items():
        if st.button(label, use_container_width=True):
            st.session_state.selected_tab = value

    st.markdown("---")

    st.header("⚙️ Settings", divider='green')

    st.subheader("🤖 Choose AI Model")
    model_options = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash",
                     "meta-llama/llama-4-maverick-17b-128e-instruct",
                     "llama-3.3-70b-versatile", "llama3-70b-8192"
                     "llama-3.1-8b-instant",
                     "deepseek-r1-distill-llama-70b",
                     "gemma2-9b-it", "gpt-4o-mini", "gpt-3.5-turbo",
                     "openai/gpt-oss-20b"]
    selected_model = st.selectbox("Model:", model_options)
    st.session_state.selected_model = selected_model

    st.subheader("🎛️ Temperature Control")
    st.caption("Adjust model creativity (Lower = More deterministic, Higher = More random)")
    temperature = st.slider("Set Temperature:", 0.0, 1.0, 0.0, 0.1)
    st.session_state.temperature = temperature

render_menu()