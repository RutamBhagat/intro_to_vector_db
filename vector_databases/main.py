import os
from tabnanny import verbose
import pinecone
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders.text import TextLoader
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone
from langchain_text_splitters import CharacterTextSplitter

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = pinecone.Pinecone(
    api_key=pinecone_api_key, environment="northamerica-northeast1-gcp"
)

if __name__ == "__main__":
    print("Hello Vectorstore")

    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "medium_blogs", "medium_blog1.txt")
    loader = TextLoader(file_path)

    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    texts = text_splitter.split_documents(document)

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-index"
    )

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        verbose=True,
    )

    query = "What is a vector DB? Give me a 15 word answer for a beginner."
    result = qa({"query": query})

    print(result)
