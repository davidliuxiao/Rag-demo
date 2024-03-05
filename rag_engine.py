import os, tempfile
import pinecone
from pathlib import Path

import PyPDF2

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
import chromadb
from langchain import OpenAI
from langchain.llms.openai import OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

import streamlit as st
# import shutil

import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

st.set_page_config(page_title="RAG")
st.title("AI in BIS test")


def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

def clean_documents():
    # shutil.rmtree(LOCAL_VECTOR_STORE_DIR.as_posix())
    # vectordb = chromadb.PersistentClient(path=LOCAL_VECTOR_STORE_DIR.as_posix(), settings=chromadb.config.Settings(anonymized_telemetry=False, allow_reset=True))
    # vectordb.reset()
    st.session_state.retriever = None
    return

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    print("local")
    vectordb = Chroma.from_documents(texts, embedding=OpenAIEmbeddings(), client=st.session_state.client,
                                     persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 6})
    return retriever

def embeddings_on_pinecone(texts):
    print("pinecone")
    pinecone.init(api_key=st.session_state.pinecone_api_key, environment=st.session_state.pinecone_env)
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
    vectordb = Pinecone.from_documents(texts, embeddings, index_name=st.session_state.pinecone_index)
    retriever = vectordb.as_retriever()
    return retriever

def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        # llm=OpenAIChat(openai_api_key=st.session_state.openai_api_key),
        llm=ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=st.session_state.openai_api_key),
        # llm=OpenAI(model_name="gpt-3.5-turbo-16k"),
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result_txt = result['answer']
    st.session_state.messages.append((query, result_txt))
    return result

def input_fields():
    print("start init")
    #
    with st.sidebar:
        #
        if "openai_api_key" in st.secrets:
            st.session_state.openai_api_key = st.secrets.openai_api_key
        else:
            st.session_state.openai_api_key = st.text_input("OpenAI API key", type="password")
        #
        if "pinecone_api_key" in st.secrets:
            st.session_state.pinecone_api_key = st.secrets.pinecone_api_key
        else: 
            st.session_state.pinecone_api_key = st.text_input("Pinecone API key", type="password")
        #
        if "pinecone_env" in st.secrets:
            st.session_state.pinecone_env = st.secrets.pinecone_env
        else:
            st.session_state.pinecone_env = st.text_input("Pinecone environment")
        #
        if "pinecone_index" in st.secrets:
            st.session_state.pinecone_index = st.secrets.pinecone_index
        else:
            st.session_state.pinecone_index = st.text_input("Pinecone index name")
    #
    # print("finish init")
    st.session_state.pinecone_db = st.toggle('Use Pinecone Vector DB')
    # print("use pinecone: " +str(st.session_state.pinecone_db))
    #
    col1, col2 = st.columns(2)

    # if hasattr(st.session_state, 'source_docs'):
    #     print("docs are1: " + str(st.session_state.source_docs))
    with col1:
        st.session_state.source_docs = st.file_uploader(label="Load Documents", type="pdf", accept_multiple_files=True)
        st.button("Learn Documents", on_click=process_documents)
    with col2:
        st.session_state.links = st.file_uploader(label="Load webpages", type="properties", accept_multiple_files=False)
        st.button("Submit links", on_click=process_links)
    # if hasattr(st.session_state, 'source_docs'):
    #     print("docs are1: " + str(st.session_state.source_docs))
    #

def prepare_doc(pdf_docs):
    docs = []
    metadata = []
    content = []

    for pdf in pdf_docs:
        print(pdf.name)
        pdf_reader = PyPDF2.PdfReader(pdf)
        for index, text in enumerate(pdf_reader.pages):
            doc_page = {'title': pdf.name + " page " + str(index + 1),
                        'content': pdf_reader.pages[index].extract_text()}
            docs.append(doc_page)
    for doc in docs:
        content.append(doc["content"])
        metadata.append({
            "title": doc["title"]
        })
    return content, metadata

def process_links():
    return

def process_documents():
    # if not st.session_state.openai_api_key or not st.session_state.pinecone_api_key or not st.session_state.pinecone_env or not st.session_state.pinecone_index or not st.session_state.source_docs:
    if not st.session_state.openai_api_key or not st.session_state.pinecone_api_key or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            clean_documents()
            st.session_state.client = chromadb.PersistentClient(path=LOCAL_VECTOR_STORE_DIR.as_posix(), settings=chromadb.config.Settings(anonymized_telemetry=False, allow_reset=True))
            st.session_state.client.reset()
            for source_doc in st.session_state.source_docs:
                #
                print("start load docs: ")
                with tempfile.NamedTemporaryFile(prefix=source_doc.name, delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                    tmp_file.write(source_doc.read())
                #
                documents = load_documents()
                print("load docs: " + str(documents))
                #
                for _file in TMP_DIR.iterdir():
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()
                #
                texts = split_documents(documents)
                print("split_documents" + str(texts))
                #
                if not st.session_state.pinecone_db:
                    st.info(f"storing in local")
                    st.session_state.retriever = embeddings_on_local_vectordb(texts)
                else:
                    st.info(f"storing in pinecone")
                    st.session_state.retriever = embeddings_on_pinecone(texts)
        except Exception as e:
            print("error: " + str(e))
            st.error(f"An error occurred: {e}")

def boot():
    #
    print("BOOT")
    input_fields()
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        col1, col2 = st.columns(2)
        with col1:
            st.chat_message("ai").write(response['answer'])
        with col2:
            st.chat_message("code").write(response['source_documents'])

if __name__ == '__main__':
    #
    boot()
    