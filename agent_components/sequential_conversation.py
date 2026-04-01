import os
from typing import List, Optional, TypedDict
from dotenv import load_dotenv, find_dotenv

# LangChain 相关
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationSummaryBufferMemory
from pydantic import BaseModel, Field

from agent_components.chromadb_file import ReadersChromadb

# LangGraph 相关
from langgraph.graph import StateGraph, START, END

load_dotenv(find_dotenv())


# --- 1. 定义 State (保持不变) ---
class State(TypedDict):
    user_input: str
    context: str
    chat_history: list
    response_obj: "ProperResponse"  # 前向引用


# --- 2. 定义 Pydantic 模型 (保持不变) ---
class ProperResponse(BaseModel):
    proper_thinking: List[str] = Field(description="针对如何回复这个问题的思考")
    final_response: str = Field(description="整理思考后的最终回复")
    worth_to_remember: bool = Field(description="从测试经验提升角度判断是否值得记忆")


class ChatTestAgentGraph:
    def __init__(self, db_path: Optional[str] = None):
        # ... (LLM 和 Memory 初始化保持不变) ...
        self.llm = ChatOpenAI(
            model="qwen3:8b",
            base_url=os.environ.get("LANGCHAIN_URL"),
            api_key="anything",
            temperature=0.7,
            tiktoken_model_name="gpt-3.5-turbo"
        )

        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=10000,
            return_messages=True,
            memory_key="chat_history",
            input_key="user_input"
        )

        # --- 这里直接复用你原来的外部工具/方法 ---
        self.vector_store = None
        if db_path:
            self.vector_store = ReadersChromadb(persist_directory=db_path)

        # 提示词和 Chain (保持不变)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("你是一个资深测试工程师。"
             "请根据以下的【参考资料】和【对话历史】进行回答。"
             "如果【参考资料】里有答案，优先依据资料回答；如果没有，请依据你的专业知识。"
             "\n\n【参考资料】:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_input}")
        ])
        self.chain = self.prompt_template | self.llm.with_structured_output(ProperResponse)

        # 构建图
        self.graph = self._build_graph()

    def _retrieve_node(self, state: State):
        """检索知识 (包装器)"""
        print("🔍 [节点] 正在调用外部工具检索...")

        #检查是否有知识库
        if not self.vector_store:
            context = "未检索到知识库"
        else:
            context = self.vector_store.search_context(user_question_str=state["user_input"])
        return {"context": context}

    def _generate_node(self, state: State):
        """生成回复"""
        print("🧠 [节点] 正在生成回复...")

        # 加载记忆
        memory_vars = self.memory.load_memory_variables({"user_input": state["user_input"]})
        history = memory_vars["chat_history"]

        # 调用 Chain
        response = self.chain.invoke({
            "context": state["context"],
            "chat_history": history,
            "user_input": state["user_input"]
        })

        return {"response_obj": response}

    def _save_memory_node(self, state: State):
        """保存记忆"""
        print("💾 [节点] 正在保存记忆...")
        self.memory.save_context(
            {"user_input": state["user_input"]},
            {"output": state["response_obj"].final_response}
        )
        return {}