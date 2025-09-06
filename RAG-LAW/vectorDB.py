import streamlit as st
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings, get_response_synthesizer
import chromadb
import Config


def init_vector_store(_nodes):
    chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection(
        name = Config.COLLECTION_NAME,
        metadata={"hnsw:space":"cosine"}
    )

    storage_context = StorageContext.from_defaults(
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    )

    if chroma_collection.count()==0 and _nodes is not None:
        storage_context.docstore.add_documents(_nodes)
        index = VectorStoreIndex(
            _nodes,
            storage_context=storage_context,
            show_progress=True
        )
        storage_context.persist(persist_dir=Config.PERSIST_DIR)
        index.storage_context.persist(persist_dir=Config.PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(
            persist_dir=Config.PERSIST_DIR,
            vector_store= ChromaVectorStore(chroma_collection=chroma_collection)
        )
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )

    return index