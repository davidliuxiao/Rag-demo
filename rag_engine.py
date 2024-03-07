import os
import pinecone
from pathlib import Path

import PyPDF2

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
import chromadb
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings


from langchain_openai import ChatOpenAI

import streamlit as st
from streamlit_js_eval import streamlit_js_eval

import time

#SeleniumURLLoader
from langchain.document_loaders import SeleniumURLLoader

#validate results
from sentence_transformers import SentenceTransformer, util


os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_API_KEY"] = 'ls__3c556b0468344b198cc40b30da61f447'

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')



st.set_page_config(page_title="RAG Demo")

with open("./static/style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


#TODO clean code
# output_parser = StrOutputParser()

def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents
# def load_documents():
#     loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
#     documents = loader.load()
#     return documents

def clean_documents():
    # shutil.rmtree(LOCAL_VECTOR_STORE_DIR.as_posix())
    # vectordb = chromadb.PersistentClient(path=LOCAL_VECTOR_STORE_DIR.as_posix(), settings=chromadb.config.Settings(anonymized_telemetry=False, allow_reset=True))
    # vectordb.reset()
    with st.spinner(text='In progress'):
        time.sleep(3)
        st.success('Knowledge Base Initialized')
    st.session_state.client.reset()
    st.session_state.retriever = None
    st.session_state.source_docs = None
    streamlit_js_eval(js_expressions="parent.window.location.reload()")
    return

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    return texts

from langchain.text_splitter import RecursiveCharacterTextSplitter
def split_links(url, title):
    cleaned_url = url.strip("b'").strip('"')
    loader = SeleniumURLLoader(urls=[cleaned_url])
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    all_splits = text_splitter.split_documents(documents)
    return all_splits
def embeddings_on_local_vectordb(texts):
    print("local")
    vectordb = Chroma.from_documents(texts, embedding=OpenAIEmbeddings(), client=st.session_state.client,
                                     persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 6})
    return retriever

# def embeddings_on_pinecone(texts):
#     print("pinecone")
#     pinecone.init(api_key=st.session_state.pinecone_api_key, environment=st.session_state.pinecone_env)
#     embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
#     vectordb = Pinecone.from_documents(texts, embeddings, index_name=st.session_state.pinecone_index)
#     retriever = vectordb.as_retriever()
#     return retriever

def query_llm(retriever, query):
    # PROMPT = PromptTemplate.from_template(_template)

    qa_chain = ConversationalRetrievalChain.from_llm(
        # llm=OpenAIChat(openai_api_key=st.session_state.openai_api_key),
        llm=ChatOpenAI(model_name=st.session_state.llm_model, openai_api_key=st.session_state.openai_api_key),
        # llm=OpenAI(model_name="gpt-3.5-turbo-16k"),
        # condense_question_prompt = PROMPT,
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
    with st.sidebar:
        st.session_state.llm_model = st.selectbox('Model', ['gpt-3.5-turbo-0125','gpt-4-0125-preview'])
        st.container(height=100, border=False)
        st.button("Reset knowledge", on_click=clean_documents)
        #
        if "openai_api_key" in st.secrets:
            st.session_state.openai_api_key = st.secrets.openai_api_key
        else:
            st.session_state.openai_api_key = st.text_input("OpenAI API key", type="password")
    #     #
    #     if "pinecone_api_key" in st.secrets:
    #         st.session_state.pinecone_api_key = st.secrets.pinecone_api_key
    #     else:
    #         st.session_state.pinecone_api_key = st.text_input("Pinecone API key", type="password")
    #     #
    #     if "pinecone_env" in st.secrets:
    #         st.session_state.pinecone_env = st.secrets.pinecone_env
    #     else:
    #         st.session_state.pinecone_env = st.text_input("Pinecone environment")
    #     #
    #     if "pinecone_index" in st.secrets:
    #         st.session_state.pinecone_index = st.secrets.pinecone_index
    #     else:
    #         st.session_state.pinecone_index = st.text_input("Pinecone index name")
    #
    # print("finish init")
    # st.session_state.pinecone_db = st.toggle('Use Pinecone Vector DB')

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


    # init Chroma DB TODO do we need pinecone ?
    st.session_state.client = chromadb.PersistentClient(path=LOCAL_VECTOR_STORE_DIR.as_posix(),
                                                        settings=chromadb.config.Settings(
                                                            anonymized_telemetry=False,
                                                            allow_reset=True))
    # if hasattr(st.session_state, 'source_docs'):
    #     print("docs are1: " + str(st.session_state.source_docs))
    #


def process_links():
    if not st.session_state.openai_api_key or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    try:
        properties = {}
        for line in st.session_state.links:
            #
            print("start load links: ")
            url, title = str(line).strip().split('|')
            properties[url.strip()] = title.strip()
            texts = split_links(url, title)
            print("split_links" + str(texts))
            #
            st.toast(f"Learning: " + title.strip().lstrip("b'").rstrip("'"))
            st.session_state.retriever = embeddings_on_local_vectordb(texts)
            # if not st.session_state.pinecone_db:
            #     st.toast(f"Learning: "+ title.strip().lstrip("b'").rstrip("'"))
            #     st.session_state.retriever = embeddings_on_local_vectordb(texts)
            # else:
            #     st.info(f"storing in pinecone")
                # st.session_state.retriever = embeddings_on_pinecone(texts)
        st.toast("Knowledge Base Updated")
    except Exception as e:
        print("error: " + str(e))
        st.error(f"An error occurred: {e}")
    return

def prepare_doc(pdf_docs):
    docs = []
    metadata = []
    content = []
    # content_metadata_list = []
    for pdf in pdf_docs:
        print(pdf.name)
        pdf_reader = PyPDF2.PdfReader(pdf)
        for index, text in enumerate(pdf_reader.pages):
            doc_page = {'title': pdf.name + " page " + str(index + 1),
                        'content': pdf_reader.pages[index].extract_text()}
            docs.append(doc_page)
    for doc in docs:
        st.toast(f"Learning: " + doc["title"])
        # content_metadata_list.append((doc["content"], {"title": doc["title"]}))
        content.append(doc["content"])
        metadata.append({
            "title": doc["title"]
        })
    return content, metadata
def get_text_chunks(content, metadata):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000,
        chunk_overlap=50,
    )
    split_docs = text_splitter.create_documents(content, metadatas=metadata)
    print(f"Documents are split into {len(split_docs)} passages")
    return split_docs

def process_documents():
    # if not st.session_state.openai_api_key or not st.session_state.pinecone_api_key or not st.session_state.pinecone_env or not st.session_state.pinecone_index or not st.session_state.source_docs:
    if not st.session_state.openai_api_key or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            print("start load docs: ")
            content, metadata = prepare_doc(st.session_state.source_docs)
            split_docs = get_text_chunks(content, metadata)
            st.session_state.retriever = embeddings_on_local_vectordb(split_docs)
            st.toast("Knowledge Base Updated")
            # with st.spinner(text='In progress'):
            #     time.sleep(1)
            #     st.success('Knowledge Base Updated')
            # if not st.session_state.pinecone_db:
            #     st.session_state.retriever = embeddings_on_local_vectordb(split_docs)
            # else:
            #     st.info(f"storing in pinecone")
                # st.session_state.retriever = embeddings_on_pinecone(texts)


            # for content, metadata in content_metadata_list:
            #     #
            #     title = metadata.get("title")
            #     #
            #     print("load docs: " + title)
            #     #
            #     texts = get_text_chunks(content, metadata)
            #     print("split_documents" + str(texts))
            #     #
            #     if not st.session_state.pinecone_db:
            #         st.info(f"Learning: "+ title)
            #         st.session_state.retriever = embeddings_on_local_vectordb(texts)
            #     else:
            #         st.info(f"storing in pinecone")
            #         st.session_state.retriever = embeddings_on_pinecone(texts)

        except Exception as e:
            print("error: " + str(e))
            st.error(f"An error occurred: {e}")

def validate_answer_against_sources(response_answer, source_documents):
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # similarity_threshold = 0.5
    # source_texts = [doc.page_content for doc in source_documents]
    #
    # answer_embedding = model.encode(response_answer, convert_to_tensor=True)
    # source_embeddings = model.encode(source_texts, convert_to_tensor=True)
    #
    # cosine_scores = util.pytorch_cos_sim(answer_embedding, source_embeddings)
    #
    # if any(score.item() > similarity_threshold for score in cosine_scores[0]):
    #     return True

    return False
# Streamed response emulator
def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def boot():
    #
    print("BOOT")
    st.image('./static/bis.png', width=100)
    st.title("BIS ChatBot")
    input_fields()
    st.image('./static/genai.png', width=80)
    st.subheader("How can I help you today?")
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if query := st.chat_input(placeholder='Message BISGPT...'):
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)

        # Post-processing step to validate the answer against the source documents
        if response['source_documents']:
            response_answer = response['answer']
            source_docs = response['source_documents']
        # is_valid_answer = validate_answer_against_sources(response_answer, source_docs)

        col1, col2 = st.columns([3, 1])
        with col1:
            # st.chat_message("ai").write(response['answer'])
            st.chat_message("ai").write_stream(response_generator(response['answer']))
        with col2:
            st.chat_message("code").write(response['source_documents'])
        # if is_valid_answer:
        #     with col2:
        #         st.chat_message("code").write(response['source_documents'])


#TODO add validation of source ref
if __name__ == '__main__':
    #
    boot()
    