import streamlit as st
import re

class Chat_Screen:
    def __init__(self):
        st.set_page_config(
            page_title="Auto BC Employment Law Consultant",
            page_icon="⚖️",
            layout="centered",
            initial_sidebar_state="auto"
        )

        self.escape_markdown_and_latex("Welcome to use AI BC Employment Law Consulting Platform, Please enter your question.")

    def disable_streamlit_watcher(self):
        """Patch Streamlit to disable file watcher"""
        def _on_script_changed(_):
            return

        from streamlit import runtime
        runtime.get_instance()._on_script_changed = _on_script_changed

    def init_chat_interface(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg.get("cleaned",msg["content"]) #use cleaned content first

            with st.chat_message(role):
                self.escape_markdown_and_latex(content)

                # If assistant message contains chain of thought
                if role == "assistant" and msg.get("think"):
                    with st.expander("Thinking(History chat"):
                        for think_content in msg["think"]:
                            self.escape_markdown_and_latex(f'<span style="color: #808080">{think_content.strip()}</span>', unsafe_allow_html=True)

                if role == "assistant" and "reference_nodes" in msg:
                    self.show_reference_details(msg["reference_nodes"])

    def show_reference_details(self, nodes):
        with st.expander("Reference nodes"):
            for idx, node in enumerate(nodes,1):
                meta = node.node.metadata
                self.escape_markdown_and_latex(f"**[{idx}] {meta['Section_Num']} {meta['Section_Name']}**")
                st.caption(f"source file: {meta['source']} | {meta['content_type']}")
                self.escape_markdown_and_latex(f"Relevancy Score: {node.score:.4f}")
                self.info_escape_markdown_and_latex(f"{node.node.text}")

    def escape_markdown_and_latex(self,text):
        special_chars = r"([*_`#&+<>$])"
        escaped_text = re.sub(special_chars, r'\\\1', text)

        return st.markdown(escaped_text)

    def info_escape_markdown_and_latex(self,text):
        special_chars = r"([*_`#&+<>$])"
        escaped_text = re.sub(special_chars, r'\\\1', text)

        return st.info(escaped_text)


