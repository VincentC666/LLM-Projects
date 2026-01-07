import streamlit as st
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

import Config


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(), verbose=True)

@st.cache_resource(show_spinner='Model initializing.....')
def init_models():

    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small"
    )

    llm = OpenAI(
        model = "gpt-4o-mini"
    )

    reranker = LLMRerank(
        top_n = Config.TOP_K,
        llm = OpenAI(model='gpt-4o-mini')
    )


    # embed_model = HuggingFaceEmbedding(
    #     model_name= Config.EMBED_MODEL_PATH
    # )

    #
    # llm = OpenAILike(
    #     model = "",
    #     api_base ="",
    #     api_key ="fake",
    #     context_window =4096,
    #     is_chat_model= True,
    #     is_function_calling_model=False
    # )

    # reranker = SentenceTransformerRerank(
    #     model = Config.RERANKER_MODEL_PATH,
    #     top_n = Config.RERANKER_TOP_N
    # )

    Settings.embed_model = embed_model
    Settings.llm = llm

    return embed_model,llm,reranker
