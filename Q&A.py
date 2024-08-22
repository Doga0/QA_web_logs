# installization with jupyter
#!pip install -q --upgrade langchain langchain_community 
#!pip install -q --upgrade chromadb

# install to run ollama
#! sudo apt-get install -y pciutils
#! curl https://ollama.ai/install.sh | sh

# import libraries
import chromadb

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain import LLMChain
from langchain_core.output_parsers import StrOutputParser


#-----Setup Ollama in colab-------
import os
import threading
import subprocess

def ollama():
    os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
    os.environ['OLLAMA_ORIGINS'] = '*'
    subprocess.Popen(["ollama", "serve"])

# Start the Ollama server
ollama_thread = threading.Thread(target=ollama)
ollama_thread.start()

# then run
#!ollama run llama3.1:8b

# start the Ollama server again
ollama_thread = threading.Thread(target=ollama)
ollama_thread.start()
#---------------------------------


# load dataset
def load_csv(file_path):
  loader = CSVLoader(file_path)
  data = loader.load()
  return data


# define the model
llm = ChatOllama(model="llama3.1:8b")

# embed dataset with embedding model and store it in vector database
def get_vectorstore(data, embedding_model):
  embedding = OllamaEmbeddings(model=embedding_model)

  vectorstore = Chroma(collection_name="logs", embedding_function=embedding)

  vectorstore.add_documents(data) 
  return vectorstore


# prepare the prompt to send llm
def prompt_func(log_data):
  prompt_template_str = """You are an assistant tasked with analyzing web traffic logs
  to extract insights and patterns. Your task is to provide a concise explanation of
  the key information within the logs, optimized for quick retrieval and analysis.
  Focus on identifying trends, anomalies, and significant events.

  Specifically, identify and explain performance issues on our users' websites,
  including bottlenecks that reduce data transfer speed, slow loading times,
  and page rendering times that place excessive load on the computer's processor.

  Log data: {log_data}

  Key Points to Extract:
  Top accessed URLs
  Unusual traffic patterns or spikes
  Common status codes and errors
  Frequent IP addresses
  Performance bottlenecks
  Time-based traffic distribution (e.g., by hour, day)
  """

  prompt_text = prompt_template_str.format(log_data=log_data)
  prompt = ChatPromptTemplate.from_template(prompt_text)
  return prompt

# generate multiple queries with MultiQueryRetriever
def retrieve_queries(vectorstore, llm, prompt, query):
  
  retriever = vectorstore.as_retriever()

  llm_chain = LLMChain(llm=llm, prompt=prompt)

  retriever_from_llm = MultiQueryRetriever(retriever=retriever,
                                  llm_chain=llm_chain)

  multi_query = retriever_from_llm.invoke(query)
  return multi_query

#  set up the RAG chain to combine everything
def rag_chain(multi_query, prompt, llm, query):
  question_runnable = RunnableLambda(lambda x: multi_query)

  retrieval = RunnableParallel({"question": question_runnable})

  rag_chain = (retrieval |
              prompt|
              llm |
              StrOutputParser())

  response = rag_chain.invoke({"question": query})

  return response


if __name__ == "__main__":

  data = load_csv(file_path='data.csv')

  vectorstore = get_vectorstore(data, "llama3.1:8b")

  log_data = "\n".join([doc.page_content for doc in data])
  prompt = prompt_func(log_data=log_data)

  query = "What are the HTTP status codes returned, and what do they indicate about the success of the requests?"
  multi_query = retrieve_queries(data, vectorstore, llm, prompt, query)

  response = rag_chain(multi_query, prompt, llm, query)

  print(response)