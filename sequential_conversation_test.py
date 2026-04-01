from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
from typing import List, Optional

# LangChain 相关导入
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationSummaryBufferMemory
from pydantic import BaseModel, Field


from agent_components.chromadb_file import ReadersChromadb

class ProperResponse(BaseModel):
    proper_thinking: List[str] = Field(description="针对如何回复这个问题的思考")
    final_response: str = Field(description="整理思考后的最终回复")
    worth_to_remember: bool = Field(description="从测试经验提升角度判断是否值得记忆")


class ChatTestAgent:
    def __init__(self, db_path: Optional[str] = None):
        # 1. 初始化 LLM
        self.llm = ChatOpenAI(
            model="qwen3:8b",
            base_url=os.environ.get("LANGCHAIN_URL"),
            api_key="anything",
            temperature=0.7,
            tiktoken_model_name="gpt-3.5-turbo"
        )

        # 2. 初始化 记忆
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=10000,
            return_messages=True,
            memory_key="chat_history",
            input_key="input"
        )

        # 3. 初始化 向量数据库 (可选)
        self.vector_store = None
        if db_path:
            self.vector_store = ReadersChromadb(persist_directory=db_path)
            print(f"✅ 知识库已加载: {db_path}")
        else:
            print("⚠️ 未加载外部知识库，仅使用模型自身能力。")

        # 4. 构建提示词模板 (加入了 context 占位符)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个资深测试工程师。"
             "请根据以下的【参考资料】和【对话历史】进行回答。"
             "如果【参考资料】里有答案，优先依据资料回答；如果没有，请依据你的专业知识。"
             "\n\n【参考资料】:\n{context}"
             ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # 5. 构建链
        self.chain = self.prompt_template | self.llm.with_structured_output(ProperResponse)

    def _search_knowledge_base(self, query: str) -> str:
        """内部方法：检索知识库"""
        return self.vector_store.search_context(user_question_str=query)

    def chat(self, user_input: str):
        """核心对话方法"""
        # --- 步骤 1: 检索知识 ---
        context = self._search_knowledge_base(user_input)

        # --- 步骤 2: 加载历史 ---
        memory_context = self.memory.load_memory_variables({"input": user_input})
        chat_history = memory_context["chat_history"]

        # --- 步骤 3: 调用模型 ---
        try:
            response = self.chain.invoke({
                "context": context,  # 注入检索到的知识
                "chat_history": chat_history,  # 注入历史记忆
                "input": user_input  # 注入用户问题
            })

            # --- 步骤 4: 保存历史 ---
            self.memory.save_context(
                {"input": user_input},
                {"output": response.final_response}
            )

            return response

        except Exception as e:
            print(f"发生错误: {e}")
            return None


# --- 主程序入口 ---
if __name__ == "__main__":

    agent = ChatTestAgent(db_path="./my_chroma_db")

    print("=== 智能测试助手启动 (输入 'quit' 退出) ===")
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        response = agent.chat(user_input)

        if response:
            print(f"\n🤖 AI 思考: {response.proper_thinking}")
            print(f"💬 AI 回复: {response.final_response}")