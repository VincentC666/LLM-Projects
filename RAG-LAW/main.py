import os
import time
import re
from llama_index.core.base.embeddings.base import similarity
from llama_index.core import get_response_synthesizer
from init_model import init_models
import streamlit as st
from load_data import load_json_files, create_nodes
from vectorDB import init_vector_store
from chat_interface import Chat_Screen
import Config
from pathlib import Path

def main():
    interface = Chat_Screen()
    interface.disable_streamlit_watcher()

    # initial chat session
    if "history" not in st.session_state:
        st.session_state.history = []

    # Load models
    embed_model, llm, reranker = init_models()

    # initial data
    if not Path(Config.VECTOR_DB_DIR).exists():
        with st.spinner("Creating Vector Database ..."):
            raw_data = load_json_files(Config.DATA_DIR)
            nodes = create_nodes(raw_data)
    else:
        nodes = None

    index = init_vector_store(nodes)
    retriver = index.as_retriever(similarity_top_k = Config.TOP_K, vector_store_query_mode="hybrid",alpha = 0.5)
    response_synthesizer = get_response_synthesizer(verbose=True)

    interface.init_chat_interface()

    if prompt := st.chat_input("Please input Employment related question"):
        # add user input to history
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            interface.escape_markdown_and_latex(prompt)

        # retrive from knowledge base
        with st.spinner("Analyzing ..."):
            start_time = time.time()

            # retrive nodes
            initial_nodes = retriver.retrieve(prompt)
            reranked_nodes = reranker.postprocess_nodes(initial_nodes,query_str=prompt)

            # filter nodes
            MIN_RERANK_SCORE = 0.4
            filtered_nodes = [node for node in reranked_nodes if node.score > MIN_RERANK_SCORE]

            if not filtered_nodes:
                response_text = ("No relevant law section were found related to your query. "
                                 "Please consider refining the inquiry or consulting a qualified attorney.")
            else:
                # generate response
                response= response_synthesizer.synthesize(prompt,nodes=filtered_nodes)
                response_text = response.response
                print(f'original response: {response_text}')

        # display response
        with st.chat_message("assistant"):

            # extract content from CoT and clean it
            think_contents = re.findall(r'<think>(.*?)</think>', response_text, re.DOTALL)
            cleaned_response = re.sub(r'<think>.*?</think>','',response_text,flags=re.DOTALL).strip()
            cleaned_response = response_text
            print(f'cleaned response: {cleaned_response}')
            # display final cleaned response
            interface.escape_markdown_and_latex(cleaned_response)

            # if CoT contents exists then display
            if think_contents:
                with st.expander("Thinking(click to open)"):
                    for content in think_contents:
                        interface.escape_markdown_and_latex(f'<span style="color: #4f7d7c">{content.strip()}</span>', unsafe_allow_html=True)

            # show the reference contents
            interface.show_reference_details(filtered_nodes[:3])

            # append assistant response to history
            st.session_state.messages.append({
                "role":"assistant",
                "content":response_text, # save original response
                "cleaned":cleaned_response, # save cleaned response
                "think":think_contents # save CoT contents
            })

if __name__ == "__main__":
    main()













