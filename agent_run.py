from langgraph.graph import StateGraph, START, END
from agent_components.sequential_conversation import ChatTestAgentGraph, State, ProperResponse


def build_and_run_agent():

    components = ChatTestAgentGraph(db_path="./my_chroma_db")

    # 2. 构建 Graph (控制逻辑)
    builder = StateGraph(State)

    builder.add_node("retrieve", lambda state: components._retrieve_node(state)) #检索知识库
    builder.add_node("generate", lambda state: components._generate_node(state)) #生成回复
    builder.add_node("save_memory", lambda state: components._save_memory_node(state)) #保存记忆
    builder.add_node("parse_api", lambda state: components._parse_api_node(state)) #分析接口定义
    builder.add_node("generate_casse", lambda state: components._generate_casse_node(state)) #生成测试用例
    builder.add_node("generate_data", lambda state: components._generate_data_node(state)) #生成测试数据
    builder.add_node("generate_assertion", lambda state: components._generate_assertion_node(state)) #生成断言规则
    builder.add_node("execute_test", lambda state: components._execute_test_node(state)) #发送 HTTP 请求
    builder.add_node("generate_report", lambda state: components._generate_report_node(state)) #生成报告

    # 连线
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "parse_api")
    builder.add_edge("parse_api", "generate_casse")
    builder.add_edge("generate_casse", "generate_data")
    builder.add_edge("generate_data", "generate_assertion")
    builder.add_edge("generate_assertion", "execute_test")
    builder.add_edge("execute_test", "generate_report")
    builder.add_edge("generate_report", "save_memory")
    builder.add_edge("save_memory", END)

    # 编译图
    graph = builder.compile()

    # 3. 封装 chat 逻辑
    def chat(user_input: str):
        initial_state = {
            "user_input": user_input,
            "original_input": user_input,
            "context": "",
            "chat_history": [],
            "response_obj": None,
            "api_definition": None,
            "test_case": None,
            "test_data": None,
            "assertion": None,
            "execution_result": None
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
        # 去除首尾空白字符（包括换行符）
        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        response = chat_func(user_input)
        if response:
            print(f"🤖 AI 思考: {response.proper_thinking}")
            print(f"💬 AI 回复: {response.final_response}")