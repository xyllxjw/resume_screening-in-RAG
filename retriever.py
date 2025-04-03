import sys
sys.dont_write_bytecode = True

from typing import List
from pydantic import BaseModel, Field

from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate
from langchain.schema.agent import AgentFinish
from langchain.tools.render import format_tool_to_openai_function


RAG_K_THRESHOLD = 5


class ApplicantID(BaseModel):
  """
  List of IDs of the applicants to retrieve resumes for
  """
  id_list: List[str] = Field(..., description="List of IDs of the applicants to retrieve resumes for")

class JobDescription(BaseModel):
  """
  Descriptions of a job to retrieve similar resumes for
  """
  job_description: str = Field(..., description="Descriptions of a job to retrieve similar resumes for") 



class RAGRetriever():
  def __init__(self, vectorstore_db, df):
    self.vectorstore = vectorstore_db
    self.df = df

  # 实现一个称为"倒数排名融合"（Reciprocal Rank Fusion，RRF）的算法，用于合并多个排序结果。
  def __reciprocal_rank_fusion__(self, document_rank_list: list[dict], k=50):
    fused_scores = {}
    for doc_list in document_rank_list:
      for rank, (doc, _) in enumerate(doc_list.items()):
        if doc not in fused_scores:
          fused_scores[doc] = 0
        fused_scores[doc] += 1 / (rank + k)
    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    return reranked_results
  
  # 这段代码的主要功能是使用向量数据库的 similarity_search_with_score 方法搜索与给定问题最相似的文档，
  # 并返回这些文档的 ID 和对应的相似度得分。
  def __retrieve_docs_id__(self, question: str, k=50):
    docs_score = self.vectorstore.similarity_search_with_score(question, k=k)
    docs_score = {str(doc.metadata["ID"]): score for doc, score in docs_score}
    return docs_score

  # 实现了一个多问题检索和重排序的功能，它将多个子问题的检索结果进行融合和重新排序
  def retrieve_id_and_rerank(self, subquestion_list: list):
    document_rank_list = []
    for subquestion in subquestion_list:
      document_rank_list.append(self.__retrieve_docs_id__(subquestion, RAG_K_THRESHOLD))
    reranked_documents = self.__reciprocal_rank_fusion__(document_rank_list)
    return reranked_documents

  #这个函数是文档检索系统的最后一步，它：
# 将排序后的文档ID转换为实际的简历内容，添加清晰的ID标识以便识别，控制返回结果的数量，保持文档的相关性排序
# 这样的实现使得系统能够返回格式良好、易于识别的简历内容，方便后续的处理和展示。
  def retrieve_documents_with_id(self, doc_id_with_score: dict, threshold=5):
    id_resume_dict = dict(zip(self.df["ID"].astype(str), self.df["Resume"]))
    retrieved_ids = list(sorted(doc_id_with_score, key=doc_id_with_score.get, reverse=True))[:threshold]
    retrieved_documents = [id_resume_dict[id] for id in retrieved_ids]
    for i in range(len(retrieved_documents)):
      retrieved_documents[i] = "Applicant ID " + retrieved_ids[i] + "\n" + retrieved_documents[i]
    return retrieved_documents 
   


class SelfQueryRetriever(RAGRetriever):
  def __init__(self, vectorstore_db, df):
    super().__init__(vectorstore_db, df)

    self.prompt = ChatPromptTemplate.from_messages([
      ("system", "You are an expert in talent acquisition."),
      ("user", "{input}")
    ])
    self.meta_data = {
      "rag_mode": "",
      "query_type": "no_retrieve",
      "extracted_input": "",
      "subquestion_list": [],
      "retrieved_docs_with_scores": []
    }
  # 这段代码定义了一个用于检索文档的嵌套函数 retrieve_docs，它接受一个用户的问题、一个 LLM 对象和一个 RAG 模式作为输入。
  def retrieve_docs(self, question: str, llm, rag_mode: str):
    # 定义了一个工具函数 retrieve_applicant_id，用于检索指定申请人的简历
    @tool(args_schema=ApplicantID)
    def retrieve_applicant_id(id_list: list):
      """Retrieve resumes for applicants in the id_list"""
      retrieved_resumes = []

      for id in id_list:
        try:
          ## 查找匹配ID的简历，使用 astype(str) 确保ID比较时类型一致，使用 iloc[0] 获取第一个匹配结果，只选择 "ID" 和 "Resume" 列
          resume_df = self.df[self.df["ID"].astype(str) == id].iloc[0][["ID", "Resume"]]
          ## 将匹配的ID和简历内容组合成一个字符串，使用换行符 \n 分隔ID和简历内容，格式为 "Applicant ID " + ID + "\n" + 简历内容
          resume_with_id = "Applicant ID " + resume_df["ID"].astype(str) + "\n" + resume_df["Resume"]
          retrieved_resumes.append(resume_with_id)
        except:
          return []
      return retrieved_resumes
    # 定义了一个工具函数 retrieve_applicant_jd，用于检索与给定职位描述最相似的简历
    @tool(args_schema=JobDescription)
    def retrieve_applicant_jd(job_description: str):
      """Retrieve similar resumes given a job description"""
      subquestion_list = [job_description]

      if rag_mode == "RAG Fusion":
        subquestion_list += llm.generate_subquestions(question)
        
      self.meta_data["subquestion_list"] = subquestion_list
      retrieved_ids = self.retrieve_id_and_rerank(subquestion_list)
      self.meta_data["retrieved_docs_with_scores"] = retrieved_ids
      retrieved_resumes = self.retrieve_documents_with_id(retrieved_ids)
      return retrieved_resumes
    
    # 这段代码定义了一个路由函数 router，用于根据 LLM 的输出选择执行不同的工具函数。
    def router(response):
      #检查响应是否为 AgentFinish类型，如果是，直接返回响应中的输出结果
      if isinstance(response, AgentFinish):
        return response.return_values["output"]
      else:
        toolbox = {
          "retrieve_applicant_id": retrieve_applicant_id,
          "retrieve_applicant_jd": retrieve_applicant_jd
        }
        self.meta_data["query_type"] = response.tool
        self.meta_data["extracted_input"] = response.tool_input
        return toolbox[response.tool].run(response.tool_input)
    

    #以下代码实现了一个 RAG (检索增强生成) 系统的核心处理链

    # 设置当前的 RAG 模式
    self.meta_data["rag_mode"] = rag_mode

    # 将工具函数转换为 OpenAI 函数格式
    llm_func_call = llm.llm.bind(functions=[format_tool_to_openai_function(tool) for tool in [retrieve_applicant_id, retrieve_applicant_jd]])

    # 创建一个链式处理流程，包括提示模板、LLM 函数调用、输出解析器和路由函数
    chain = self.prompt | llm_func_call | OpenAIFunctionsAgentOutputParser() | router

    # 执行整个处理流程，传入用户的问题作为输入
    result = chain.invoke({"input": question})

    return result
