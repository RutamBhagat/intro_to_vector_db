import os

import pinecone
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders.text import TextLoader
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone
from langchain_text_splitters import CharacterTextSplitter


if __name__ == "__main__":
    print("Hello Vectorstore")
