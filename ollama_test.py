from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,  # 人类用户输入的消息
    AIMessage,  # AI 模型生成的回复
    SystemMessage,  # 系统级指令或角色设定
    ToolMessage,  # 工具（Tool）调用后返回的结果
)

from pydantic import BaseModel, Field

from langchain_classic.memory import ConversationSummaryBufferMemory
#from langchain_community.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(
    model="qwen3:8b",
    base_url=os.environ.get("LANGCHAIN_URL"),
    api_key="anything",
    temperature=0.7,
    tiktoken_model_name="gpt-3.5-turbo"
)


class ProperResponse(BaseModel):
    proper_thinking: list[str] = Field(
        description="针对如何回复这个问题的思考"
    )
    final_response: str = Field(
        description="整理思考后的最终回复"
    )
    worth_to_remember: bool = Field(
        description="从测试经验提升角度判断是否值得记忆"
    )


memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,
    return_messages=True,
    memory_key="chat_history",
    input_key="input"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个资深测试工程师。请根据以下的对话历史和用户最新输入进行回答。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt_template | llm.with_structured_output(ProperResponse)


def run_conversation():
    print("=== 多轮对话测试开始 (输入 'quit' 退出) ===")

    # 模拟多轮对话循环
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        # 获取当前的记忆内容 (包含摘要和近期对话)
        # load_memory_variables 会根据 max_token_limit 自动决定是否生成摘要
        memory_context = memory.load_memory_variables({"input": user_input})
        chat_history = memory_context["chat_history"]

        # 调试信息：查看当前传入模型的上下文长度和是否有摘要
        # 在实际生产中可以去掉
        total_chars = sum(len(str(msg.content)) for msg in chat_history)
        has_summary = any("总结" in str(msg.content) or "summary" in str(msg.content).lower() for msg in chat_history if
                          isinstance(msg, SystemMessage))
        print(
            f"[DEBUG] 当前上下文消息数: {len(chat_history)}, 估算字符数: {total_chars}, 是否已触发摘要: {has_summary}")

        try:
            # 调用模型
            response = chain.invoke({
                "chat_history": chat_history,
                "input": user_input
            })

            # 打印结构化结果
            print(f"\nAI 思考: {response.proper_thinking}")
            print(f"AI 回复: {response.final_response}")
            print(f"值得记忆: {response.worth_to_remember}")

            # 【关键步骤】将本轮对话存入记忆
            # 必须保存 HumanMessage 和 AIMessage
            memory.save_context(
                {"input": user_input},
                {"output": response.final_response}  # 这里只存最终回复文本作为历史
            )

        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == "__main__":
    run_conversation()