from langgraph.graph import StateGraph, START, END
from agent_components.sequential_conversation import ChatTestAgentGraph, State, ProperResponse


def build_and_run_agent():

    components = ChatTestAgentGraph(db_path="./my_chroma_db")

    # 2. 构建 Graph (控制逻辑)
    builder = StateGraph(State)

    builder.add_node("retrieve", components._retrieve_node)
    builder.add_node("generate", components._generate_node)
    builder.add_node("save_memory", components._save_memory_node)

    # 连线
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "save_memory")
    builder.add_edge("save_memory", END)

    # 编译图
    graph = builder.compile()

    # 3. 封装 chat 逻辑
    def chat(user_input: str):
        initial_state = {
            "user_input": user_input,
            "context": "",
            "chat_history": [],
            "response_obj": None
        }
        result = graph.invoke(initial_state)
        return result["response_obj"]

    return chat


# --- 主程序入口 ---
if __name__ == "__main__":
    chat_func = build_and_run_agent()

    print("=== 智能测试助手启动 (输入 'quit' 退出) ===")
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        response = chat_func(user_input)
        if response:
            print(f"🤖 AI 思考: {response.proper_thinking}")
            print(f"💬 AI 回复: {response.final_response}")