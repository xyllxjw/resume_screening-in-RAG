import sys, os
sys.dont_write_bytecode = True

import time
from dotenv import load_dotenv

import pandas as pd
import streamlit as st
import openai
from streamlit_modal import Modal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from llm_agent import ChatBot
from ingest_data import ingest
from retriever import SelfQueryRetriever
import chatbot_verbosity as chatbot_verbosity

#
import tkinter as tk
from tkinter import filedialog

# # 创建一个隐藏的主窗口，存放文件选择对话框
root = tk.Tk()
root.withdraw()# 隐藏主窗口


from pypdf import PdfReader
import glob
import csv

# 上传pdf文件时，转换为csv文件的保存路径
OUT_PATH_FILE = "./data/resume-data/pdf-resumes.csv"

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")                # 默认数据路径./data/resume-data/resumes.csv
FAISS_PATH = os.getenv("FAISS_PATH")              # 向量数据库路径./data/resume-data/faiss_index
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")    # 向量模型sentence-transformers/all-MiniLM-L6-v2

welcome_message = """
  #### Introduction 🚀

  The system is a RAG pipeline designed to assist hiring managers in searching for the most suitable candidates out of thousands of resumes more effectively. ⚡

  The idea is to use a similarity retriever to identify the most suitable applicants with job descriptions.
  This data is then augmented into an LLM generator for downstream tasks such as analysis, summarization, and decision-making. 

  #### Getting started 🛠️

  1. To set up, please add your OpenAI's API key. 🔑 
  2. Type in a job description query. 💬

  Hint: The knowledge base of the LLM has been loaded with a pre-existing vectorstore of [resumes](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/blob/main/data/main-data/synthetic-resumes.csv) to be used right away. 
  In addition, you may also find example job descriptions to test [here](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/blob/main/data/supplementary-data/job_title_des.csv).

  Please make sure to check the sidebar for more useful information. 💡
"""

info_message = """
  # Information

  ### 1. What if I want to use my own resumes?

  If you want to load in your own resumes file, simply use the uploading button above. 
  Please make sure to have the following column names: `Resume` and `ID`. 

  Keep in mind that the indexing process can take **quite some time** to complete. ⌛

  ### 2. What if I want to set my own parameters?

  You can change the RAG mode and the GPT's model type using the sidebar options above. 

  About the other parameters such as the generator's *temperature* or retriever's *top-K*, I don't want to allow modifying them for the time being to avoid certain problems. 
  FYI, the temperature is currently set at `0.1` and the top-K is set at `5`.  

  ### 3. Is my uploaded data safe? 

  Your data is not being stored anyhow by the program. Everything is recorded in a Streamlit session state and will be removed once you refresh the app. 

  However, it must be mentioned that the **uploaded data will be processed directly by OpenAI's GPT**, which I do not have control over. 
  As such, it is highly recommended to use the default synthetic resumes provided by the program. 

  ### 4. How does the chatbot work? 

  The Chatbot works a bit differently to the original structure proposed in the paper so that it is more usable in practical use cases.

  For example, the system classifies the intent of every single user prompt to know whether it is appropriate to toggle RAG retrieval on/off. 
  The system also records the chat history and chooses to use it in certain cases, allowing users to ask follow-up questions or tasks on the retrieved resumes.
"""

about_message = """
  # About

  This small program is a prototype designed out of pure interest as additional work for the author's Bachelor's thesis project. 
  The aim of the project is to propose and prove the effectiveness of RAG-based models in resume screening, thus inspiring more research into this field.

  The program is very much a work in progress. I really appreciate any contribution or feedback on [GitHub](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline).

  If you are interested, please don't hesitate to give me a star. ⭐
"""

st.set_page_config(page_title="Resume Screening in RAG")
st.title("Resume Screening in RAG")

# Initialize session state--------------------------------
#历史对话为空，则显示欢迎消息
if "chat_history" not in st.session_state:
  st.session_state.chat_history = [AIMessage(content=welcome_message)]

# df 为空，则读取默认数据到df，也就是已经放在目录中的resumes.csv文件
if "df" not in st.session_state:
  st.session_state.df = None

# 设置embedding模型
if "embedding_model" not in st.session_state:
  st.session_state.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})

# 如果RAG pipeline为空，则加载向量数据库FAISS_PATH中的数据
# 修改这个逻辑，如果为空，则先定义一个空值，然后上传文件后，向量化，存向量数据库一套搞完后，再初始化
if "rag_pipeline" not in st.session_state:
  st.session_state.rag_pipeline = None

#   vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
#   st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)

# 先设置resume list为空，这是用来存放检索到的简历列表
if "resume_list" not in st.session_state:
  st.session_state.resume_list = []

# 选择pdf上传时，选择文件夹初始为空
if "uploaded_pdf_file_path" not in st.session_state:
  st.session_state.uploaded_pdf_file_path = None

if "uploaded" not in st.session_state:
  st.session_state.uploaded = None

# if "reload" not in st.session_state:
#   st.session_state.reload = False

# 将pdf文件转换为csv文件，存放路径为OUT_PATH_FILE
def convert_pdf_to_csv(pdfs):
  # 检查OUT_PATH_FILE是否存在，如果存在则删除
  if os.path.exists(OUT_PATH_FILE):
     os.remove(OUT_PATH_FILE)

  id = 1
  with open(OUT_PATH_FILE, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)    
    writer.writerow(["ID", "Resume"])

    for pdf_path in pdfs:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        writer.writerow([id, text])
        id += 1  
  # return OUT_PATH_FILE

#上传csv文件，进行向量化，ingest数据
def upload_file():
  modal = Modal(key="Demo Key", title="File Error", max_width=500)
  if st.session_state.uploaded_csv_file != None:
    try:  
      df_load = pd.read_csv(st.session_state.uploaded_csv_file)
    except Exception as error:
      with modal.container():
        st.markdown("The uploaded file returns the following error message. Please check your csv file again.")
        st.error(error)
    else:
      if "Resume" not in df_load.columns or "ID" not in df_load.columns:
        with modal.container():
          st.error("Please include the following columns in your data: \"Resume\", \"ID\".")
      else:
        with st.toast('Indexing the uploaded data. This may take a while...'):
          st.session_state.df = df_load
          print("df是:",st.session_state.df)
          # 将读取到的cvs文件中的Resume列分块，并生成向量数据库，存入FAISS_PATH中
          vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
          # 设置RAG pipeline
          # st.session_state.retriever = SelfQueryRetriever(vectordb, st.session_state.df)
          st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)
  else:
    # 如果上传的文件为空，则读取默认数据到df，也就是已经放在目录中的resumes.csv文件 
    st.session_state.df = pd.read_csv(DATA_PATH)
    print("df1:",st.session_state.df)
    # 加载向量数据库FAISS_PATH中的数据
    vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
    # 设置 RAG pipeline
    st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)


def check_openai_api_key(api_key: str):
  openai.api_key = api_key
  try:
    _ = openai.chat.completions.create(
      model="gpt-4o-mini",  # Use a model you have access to
      messages=[{"role": "user", "content": "Hello!"}],
      max_tokens=3
    )
    return True
  except openai.AuthenticationError as e:
    return False
  else:
    return True
  
  
def check_model_name(model_name: str, api_key: str):
  openai.api_key = api_key
  model_list = [model.id for model in openai.models.list()]
  return True if model_name in model_list else False


def clear_message():
  st.session_state.resume_list = []
  st.session_state.chat_history = [AIMessage(content=welcome_message)]

def reset_selectbox():
    del st.session_state.uploaded

user_query = st.chat_input("Type your message here...")

with st.sidebar:
  st.markdown("# Control Panel")

  st.text_input("OpenAI's API Key", type="password", key="api_key")
  st.selectbox("RAG Mode", ["Generic RAG", "RAG Fusion"], placeholder="Generic RAG", key="rag_selection")
  st.text_input("GPT Model", "gpt-4o-mini", key="gpt_selection")
  # st.file_uploader("Upload resumes", type=["csv"], key="uploaded_csv_file", on_change=upload_file)

  # 创建下拉选项框
  st.selectbox(
      'select the format of resumes：',
      ('please select,the pdf file will be converted to csv file', 'Upload resumes-pdf', 'Upload resumes-cvs'),
      key="uploaded",
      # index=0
  )
  print("uploaded:",st.session_state.uploaded)
  # 根据选择显示文件上传框
  
      # st.session_state.uploaded = False
      # Make folder picker dialog appear on top of other windows
  if st.session_state.uploaded == 'Upload resumes-pdf':
      root.wm_attributes('-topmost', 1)
      st.session_state.uploaded_pdf_file_path = filedialog.askdirectory(title="choose a folder",master=root)
        
      if st.session_state.uploaded_pdf_file_path is not None:
        try:
          dir_path = f"{st.session_state.uploaded_pdf_file_path}/*.pdf"         
          pdfs = glob.glob(dir_path)
          convert_pdf_to_csv(pdfs)            
          st.session_state.uploaded_csv_file = OUT_PATH_FILE
          upload_file()
              
        except Exception as e:
          st.error(f"upload pdf file error: {e}")
            
        st.success("PDF Resumes already uploaded,you can choose other resumes upload, to switch the base reusmes data")
       
      
  elif st.session_state.uploaded == 'Upload resumes-cvs':      
      st.file_uploader("Upload resumes in csv format", type=["csv"], key="uploaded_csv_file", on_change=upload_file)        
      # st.session_state.uploaded = 'Upload resumes-cvs'  # 标记文件已上传
      st.success("CSVResumes already uploaded,you can choose other resumes upload, to switch the base reusmes data")     
      
  print("index1:",st.session_state.uploaded)
  
  
  st.button("Clear conversation", on_click=clear_message)

  st.divider()
  st.markdown(info_message)

  st.divider()
  st.markdown(about_message)
  st.markdown("this is a resume screening in RAG for testing")


for message in st.session_state.chat_history:
  if isinstance(message, AIMessage):
    with st.chat_message("AI"):
      st.write(message.content)
  elif isinstance(message, HumanMessage):
    with st.chat_message("Human"):
      st.write(message.content)
  else:
    with st.chat_message("AI"):
      message[0].render(*message[1:]) #调用 message[0] 对象的 render 方法，并将 message[1:] 中的所有元素作为参数传递给这个方法


if not st.session_state.api_key:
  st.info("Please add your OpenAI API key to continue. ")
  st.stop()

if not check_openai_api_key(st.session_state.api_key):
  st.error("The API key is incorrect. Please set a valid OpenAI API key to continue. ")
  st.stop()

if not check_model_name(st.session_state.gpt_selection, st.session_state.api_key):
  st.error("The model you specified does not exist. ")
  st.stop()


retriever = st.session_state.rag_pipeline

# 初始化ChatBot类用于
llm = ChatBot(
  api_key=st.session_state.api_key,
  model=st.session_state.gpt_selection,
)

if user_query is not None and user_query != "": #确保 user_query 变量有有效的值
  with st.chat_message("Human"):
    st.markdown(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))

  with st.chat_message("AI"):
    start = time.time()
    with st.spinner("Generating answers..."):
      #这里开始调用retriever.py的retrieve_docs方法，执行链式调用
      document_list = retriever.retrieve_docs(user_query, llm, st.session_state.rag_selection)
      # 获取检索类型
      query_type = retriever.meta_data["query_type"]
      # 将检索到的简历存储到resume_list中
      st.session_state.resume_list = document_list
      # 调用llm.generate_message_stream方法，生成响应
      stream_message = llm.generate_message_stream(user_query, document_list, [], query_type)
    end = time.time()
    
    # 将响应写入到页面中
    response = st.write_stream(stream_message)
    # 调用chatbot_verbosity.py的render方法，渲染响应  
    retriever_message = chatbot_verbosity
    retriever_message.render(document_list, retriever.meta_data, end-start)

    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.chat_history.append((retriever_message, document_list, retriever.meta_data, end-start))