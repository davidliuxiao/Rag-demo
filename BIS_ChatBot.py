import os
from pathlib import Path

import PyPDF2

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings

#TODO uncomment before deploying, to overcome streamlit bug
import sys
# sys.modules['sqlite3'] = __import__('pysqlite3')
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import chromadb
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings


from langchain_text_splitters import HTMLHeaderTextSplitter

from langchain_openai import ChatOpenAI

import streamlit as st
from streamlit_js_eval import streamlit_js_eval



import time
import base64


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter



os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_API_KEY"] = 'ls__3c556b0468344b198cc40b30da61f447'
os.environ["ALLOW_RESET"] = 'TRUE'

LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')



st.set_page_config(page_title="BIS ChatBot")

with open("./static/style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

PDF_DIR = Path(__file__).resolve().parent.joinpath('data', 'pdfs')

# Create a directory to save uploaded files
# PDF_DIR = Path('/').joinpath('usr', 'share', 'nginx', 'data', 'pdfs')
PDF_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).resolve().parent.joinpath('static')

def load_documents():
    loader = DirectoryLoader(PDF_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents


def delete_files(path):
    for filename in os.listdir(path):
        if os.path.isfile(os.path.join(path, filename)):
            os.remove(os.path.join(path, filename))
            print("File removed :" + os.path.join(path, filename))



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
    # TODO delete all pdfs
    streamlit_js_eval(js_expressions="parent.window.location.reload()")

    # Clean pdfs uploaded
    delete_files(PDF_DIR.as_posix())
    return

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap)
    texts = text_splitter.split_documents(documents)
    return texts


def get_text_chunks(content, metadata):
    text_splitter = st.session_state.text_splitter
    split_docs = text_splitter.create_documents(content, metadatas=metadata)
    print(f"Documents are split into {len(split_docs)} passages")
    return split_docs

from langchain.text_splitter import RecursiveCharacterTextSplitter


# def split_links2(url):
#     cleaned_url = url.strip("b'").strip('"')
#     loader = SeleniumURLLoader(urls=[cleaned_url])
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap)
#     all_splits = text_splitter.split_documents(documents)
#     return all_splits

def split_links(url):
    cleaned_url = url.strip("b'").strip('"')
    headers_to_split_on = [
        ("h1", "title"),
        ("h3", "Content")
    ]
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    html_header_splits = html_splitter.split_text_from_url(cleaned_url)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap)
    all_splits = text_splitter.split_documents(html_header_splits)
    #Filter splits
    filtered_splits = []
    for split in all_splits:
        if split.metadata.get('title') is not None and split.metadata.get('Content') is not None:
            split.metadata['source'] = url
            split.metadata['detail'] = split.page_content
            filtered_splits.append(split)
    return filtered_splits

def embeddings_on_local_vectordb(texts):
    print("local")
    st.session_state.embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
    vectordb = Chroma.from_documents(texts, embedding=st.session_state.embeddings, client=st.session_state.client,
                                     persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    st.session_state.retriever = vectordb.as_retriever()
    return get_retriever()


def get_retriever():
    splitter = st.session_state.text_splitter
    redundant_filter = EmbeddingsRedundantFilter(embeddings=st.session_state.embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=st.session_state.embeddings,
                                       similarity_threshold=st.session_state.docs_similarity)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=st.session_state.retriever
    )
    return compression_retriever


def query_llm(query):
    retriever = get_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=st.session_state.llm_model, openai_api_key=st.session_state.openai_api_key),
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


    col1, col2 = st.columns(2)

    with col1:
        st.session_state.source_docs = st.file_uploader(label="Load Documents", type="pdf", accept_multiple_files=True)
        st.button("Learn Documents", on_click=process_documents)
    with col2:
        st.session_state.links = st.file_uploader(label="Load webpages", type="properties", accept_multiple_files=False)
        st.button("Learn urls", on_click=process_links)


def init_sidebar():
    with st.sidebar:
        st.session_state.llm_model = st.selectbox('Model', ['gpt-3.5-turbo-0125', 'gpt-4-0125-preview'])
        st.container(height=30, border=False)
        st.session_state.docs_similarity = st.slider("Documents Relevance", float(0.0), float(1.0), float(0.72),help='Relevance level for source inclusion')

        st.container(height=30, border=False)
        st.session_state.chunk_size = 2000
        st.session_state.chunk_size = st.slider("Chunk Size", 100, 100000, 30000, help='Unit of information provide to context')
        # st.session_state.chunk_overlap = 50
        # st.session_state.chunk_overlap = st.slider("Chunk Overlap", 0, 2000, 100, help='Unit of information provide to context')
        st.session_state.chunk_overlap = 100

        #
        if "openai_api_key" in st.secrets:
            st.session_state.openai_api_key = st.secrets.openai_api_key
        else:
            st.session_state.openai_api_key = st.text_input("OpenAI API key", type="password")

        if "retriever" in st.session_state:
            st.container(height=80, border=False)
            st.button("Reset knowledge", on_click=clean_documents)

        st.container(height=50, border=False)
        url = "https://smith.langchain.com/"
        title = "MLOps Monitoring"
        st.markdown(
            """<a href="https://smith.langchain.com/">
            <img src="app/static/mlops.png" width="50">
            </a>
            """
            ,
            unsafe_allow_html=True,
        )


def init_states():

    st.session_state.links = {}
    st.session_state.source_docs = {}
    st.session_state.client = chromadb.PersistentClient(path=LOCAL_VECTOR_STORE_DIR.as_posix())

    # st.session_state.retriever = Chroma(embedding_function=st.session_state.embeddings, client=st.session_state.client,
    #                                  persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix()).as_retriever()


def process_links():
    if not st.session_state.openai_api_key or not st.session_state.links:
        st.warning(f"Please upload the urls and provide the missing fields in the side-bar")
    else:
        try:
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state.chunk_size,
                                                                            chunk_overlap=st.session_state.chunk_overlap)
            # properties = {}
            for line in st.session_state.links:
                #
                url = str(line).strip().lstrip("b'").strip("\\r\\n'")
                texts = split_links(url)
                print("split_urls" + str(texts))
                #
                st.toast(f"Learning: " + url.strip().lstrip("b'").rstrip("'"))
                st.session_state.retriever = embeddings_on_local_vectordb(texts)
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
            doc_page = {'title': pdf.name,
                        'page' : str(index + 1),
                        'content': pdf_reader.pages[index].extract_text()}
            docs.append(doc_page)
    for doc in docs:
        st.toast(f"Learning: " + doc["title"] + ': page '+ doc["page"])
        # content_metadata_list.append((doc["content"], {"title": doc["title"]}))
        content.append(doc["content"])
        metadata.append({
            "title": doc["title"],
            "page": doc["page"],
            "type" :"pdf"
        })
    return content, metadata


def save_pdf(source_docs):
    for source_doc in source_docs:
        with open(os.path.join(PDF_DIR.as_posix(), source_doc.name), "wb") as f:
            f.write(source_doc.getbuffer())
        print("Saved File: " + source_doc.name)


def process_documents():
    if not st.session_state.openai_api_key or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields in the side-bar")
    else:
        try:
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state.chunk_size,
                                                                            chunk_overlap=st.session_state.chunk_overlap)
            print("start load docs: ")
            save_pdf(st.session_state.source_docs)
            content, metadata = prepare_doc(st.session_state.source_docs)
            split_docs = get_text_chunks(content, metadata)
            st.session_state.retriever = embeddings_on_local_vectordb(split_docs)
            st.toast("Knowledge Base Updated")

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

def format_source_doc(source_docs):
    source_docs_formatted = []
    for doc in source_docs:
        if doc.metadata.get("page") is not None and int(doc.metadata.get("page")) > 0:
            title =  doc.metadata.get("title") + ' page '+doc.metadata.get("page")
            #pdf link
            source = 'http://localhost/data/pdfs/' + doc.metadata.get("title")

            detail = doc.page_content
            similarity_score = doc.state['query_similarity_score']
            doc_formatted = {'title': title,'source': source, 'detail': detail, 'relevance': similarity_score}
            # doc_formatted = {'title': title,'source': pdf_display, 'detail': detail}
        else:
            title =  doc.metadata.get("title")
            source = doc.metadata.get("source")
            detail = doc.page_content
            similarity_score = doc.state['query_similarity_score']
            doc_formatted = {'title': title,'source': source, 'detail': detail, 'relevance': similarity_score}
        source_docs_formatted.append(doc_formatted)
    return source_docs_formatted

def boot():
    #
    print("BOOT")
    st.image('./static/bis.png', width=80)
    st.header("BIS ChatBot", anchor=False, divider='gray')
    init_sidebar()
    init_states()
    input_fields()
    st.image('./static/genai.png', width=50)
    st.subheader("How can I help you today?", anchor=False)
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #
    for message in st.session_state.messages:
        st.chat_message('human', avatar='./static/user.png').write(message[0])
        st.chat_message('ai', avatar='./static/genai_red.png').write(message[1])
    #
    if query := st.chat_input(placeholder='Message BISGPT...'):
        st.chat_message("human", avatar='./static/user.png').write(query)
        if "retriever" not in st.session_state:
            st.warning("Please upload the documents or urls first")
        else:
            response = query_llm(query)

            # Post-processing step to validate the answer against the source documents
            if response['source_documents']:
                response_answer = response['answer']
                source_docs = response['source_documents']
            # is_valid_answer = validate_answer_against_sources(response_answer, source_docs)
            source_docs_formatted =format_source_doc(response['source_documents'])
            col1, col2 = st.columns([3, 1])
            with col1:
                # st.chat_message("ai").write(response['answer'])
                st.chat_message("ai", avatar='./static/genai_red.png').write_stream(response_generator(response['answer']))
            with col2:
                with st.chat_message("source_documents"):
                    for doc in source_docs_formatted:
                        title = doc['title']
                        url = doc['source'].strip("b'").strip('"')
                        link = f'[{title}]({url})'
                        st.markdown(link, help='relevance:' + str(doc['relevance']) + '  ' +doc['detail'])


if __name__ == '__main__':
    #
    boot()
    