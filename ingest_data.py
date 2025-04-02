import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import DataFrameLoader

DATA_PATH = "../data/main-data/synthetic-resumes.csv"
FAISS_PATH = "../vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def ingest(df: pd.DataFrame, content_column: str, embedding_model):
  # 这段代码的主要功能是创建一个 DataFrameLoader 的实例，使用给定的 Pandas 数据框 df 和指定的内容列 content_column。这个实例 loader 将用于后续的数据处理操作，例如加载、转换或分析数据框中的文本内容。
  loader = DataFrameLoader(df, page_content_column=content_column)

  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1024,
    chunk_overlap = 500
  )

  documents = loader.load()
  document_chunks = text_splitter.split_documents(documents)

  #调用 FAISS 类的 from_documents 方法，将分割后的文档块转换为向量并存储在 FAISS 数据库中。
  vectorstore_db = FAISS.from_documents(document_chunks, embedding_model, distance_strategy=DistanceStrategy.COSINE)
  #vectorstore_db.save_local(FAISS_PATH)
  return vectorstore_db