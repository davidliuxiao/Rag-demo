import logging
import time
# logging.basicConfig(
#     format='%(asctime)s %(levelname)-8s %(message)s',
#     level=logging.INFO,
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '***'
os.environ["OPENAI_API_KEY"] = 'sk-***'

# Use openai API
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# llm = OpenAI(model_name="gpt-4")
llm = OpenAI(model_name="gpt-3.5-turbo-16k")

#SeleniumURLLoader
from langchain.document_loaders import SeleniumURLLoader
urls = [
    "https://lde.tbe.taleo.net/lde01/ats/careers/requisition.jsp?org=BIS&cws=1&rid=1167",
    "https://lde.tbe.taleo.net/lde01/ats/careers/requisition.jsp?org=BIS&cws=1&rid=1168",
    "https://lde.tbe.taleo.net/lde01/ats/careers/requisition.jsp?org=BIS&cws=1&rid=1164"
]
loader = SeleniumURLLoader(urls=urls)
documents = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
all_splits = text_splitter.split_documents(documents)


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

# embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# storing embeddings in the vector store
vectorstore1 = FAISS.from_documents(all_splits, embeddings)
vectorstore2 = FAISS.from_documents(all_splits, embeddings)

vectorstore1.merge_from(vectorstore2)


from langchain.chains import ConversationalRetrievalChain

chain = ConversationalRetrievalChain.from_llm(llm, vectorstore1.as_retriever(), return_source_documents=True, return_generated_question =True)
#Does it pass previous Retrieval in the following questions ?


# QA with loader1
chat_history = []

# query = "What is Data lakehouse architecture in Databricks?"
# query = "Give a detailed summarize BIS covid topic"
query = "Give a summary of the following roles in BIS current vacancies and their application deadline, print in a table format :  Senior Security Specialist,  Payroll Manager, IT Infrastructure Operations Analyst"
docs = vectorstore1.similarity_search(query)
print("Fetched %s docs as retrievals" %(len(docs)))

print("Prompt details are : ")
print(chain.combine_docs_chain)

from langchain.callbacks import get_openai_callback
with get_openai_callback() as cb:
    result = chain({"question": query, "chat_history": chat_history})
    print(cb)

#TODO create a dict of url - title mapping to print
print("Answer is : ")
print(result['answer'])

from utils.knowledges import PropertiesReader
file_path = 'data/urls.properties'  # Specify the path to your properties file
properties_reader = PropertiesReader(file_path)

print("References : ")
references = {}
for item in result['source_documents']:
    metadata = item.metadata
    source = metadata.get("source")
    title = properties_reader.get_property(source)
    if title:
        references[source]=title

for key, value in references.items():
    print(f"{key}: {value}")


print(result['source_documents'])