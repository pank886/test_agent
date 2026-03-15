from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os

from langchain_openai import ChatOpenAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

#向量化用
# embeddings = OllamaEmbeddings(
#     model="m3e-base:latest",
#     base_url=os.environ.get("EMBEDDING_URL")
# )
class ReadersChromadb:
    def __init__(
            self,
            persist_directory="./my_chroma_db",
            collection_name="my_rag_collection",
            embedding_model_name=None,
            base_url=None
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # 如果没有传入 base_url，尝试从环境变量获取
        if base_url is None:
            from dotenv import load_dotenv, find_dotenv
            load_dotenv(find_dotenv())
            base_url = os.environ.get("EMBEDDING_URL")

        # ✅ 内部创建 embeddings 实例，不再依赖全局变量
        self.embeddings = OllamaEmbeddings(
            model=embedding_model_name,
            base_url=base_url
        )

        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        print(f"✅ 向量数据库已初始化 (模型: {embedding_model_name})")

    def extract_text_from_pdf(self,pdf):
        """
        提取PDF中的文本
        :param pdf:
        :return:
        """
        with open(pdf, "rb") as file:
            reader = PdfReader(file)
            if reader.metadata:
                print("标题：", reader.metadata.title)
            print("文档总页数：", len(reader.pages))
            full_text = "\n\n".join([page.extract_text() for page in reader.pages])
        return full_text

    def chunk_text(self, full_text):
        """
        文本切块
        :param full_text:
        :return:
        """
        api_splitter = RecursiveCharacterTextSplitter(
            # 1. 块大小
            chunk_size=1000,

            # 2. 重叠长度
            chunk_overlap=200,

            # 3. 分隔符策略
            separators=[
                "\n## ",  # Markdown 二级标题 (常见于接口名称)
                "\n### ",  # Markdown 三级标题 (常见于 "请求参数", "返回示例")
                "\n---\n",  # 水平分割线 (很多文档用来分隔不同接口)
                "\n\n",  # 双换行 (段落间隔)
                "\n",  # 单换行 (行间隔)
                " ",  # 空格
                ""  # 字符
            ],

            length_function=len  # 长度控制单位
        )

        chunks = api_splitter.split_text(full_text)
        return chunks

    def vectorization(self, chunks):
        """
        文本向量化
        :param chunks:
        :return:
        """
        documents = [Document(page_content=chunk) for chunk in chunks]

        print(f"正在将 {len(documents)} 个文本块向量化并存储...")

        self.vector_store.add_documents(documents=documents)

        print("✅ 向量化完成！数据已存入 Chroma 数据库。")

    def search_context(self, user_question_str):

        results = self.vector_store.similarity_search(user_question_str, k=5)

        # 提取内容返回给 LLM
        context_contents = [doc.page_content for doc in results]
        return "\n\n".join(context_contents)


if __name__ == "__main__":
    reader = ReadersChromadb()
    pdftext = reader.extract_text_from_pdf(r'D:\coffee_swagger.pdf')
    chunks = reader.chunk_text(pdftext)
    reader.vectorization(chunks)
    ans = reader.search_context(user_question_str="订单接口文档")
    print("文档内容_全：", ans)
