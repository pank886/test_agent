import json
import os
from datetime import datetime
from typing import Optional, TypedDict

import httpx
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate

# LangChain 相关
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationSummaryBufferMemory

from agent_components.chromadb_file import ReadersChromadb
from prompts.response_model import ProperResponse, ApiDefinition, TestCase, TestData, AssertionRule, ExecutionResult, TestReport
from prompts.definitions import PromptFactory

load_dotenv(find_dotenv())


# ---  定义 State  ---
class State(TypedDict):
    user_input: str
    original_input: str  # 专门用来存第一次的输入
    context: str
    chat_history: list
    response_obj: "ProperResponse"

    api_definition: Optional[ApiDefinition]
    test_case: Optional[TestCase]
    test_data: Optional[TestData]
    assertion: Optional[AssertionRule]
    execution_result: Optional[ExecutionResult]


class ChatTestAgentGraph:
    def __init__(self, db_path: Optional[str] = None):
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

        self.prompt_factory = PromptFactory()

        self.vector_store = None
        if db_path:
            self.vector_store = ReadersChromadb(persist_directory=db_path)

        # 提示词和 Chain
        prompt_template = self.prompt_factory.get_prompt_template()
        self.chain = prompt_template | self.llm.with_structured_output(ProperResponse, strict=False)

    def _retrieve_node(self, state: State):
        """检索知识库 (包装器)"""
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
        """保存完整测试报告到文件,直接序列化 State"""
        print("💾 [节点] 正在持久化完整测试数据...")

        # 1. 准备保存的数据目录
        save_dir = "test_history"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 2. 将 Pydantic 模型转换为字典 (处理 datetime 等特殊类型)
        def serialize(obj):
            if hasattr(obj, 'model_dump'): # Pydantic V2
                return obj.model_dump()
            return str(obj)

        # 3. 提取关键数据
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "user_input": state.get("user_input"),
            "api_definition": serialize(state.get("api_definition")),
            "test_case": serialize(state.get("test_case")),
            "test_data": serialize(state.get("test_data")),
            "assertion": serialize(state.get("assertion")),
            "execution_result": serialize(state.get("execution_result"))
        }

        # 4. 写入 JSON 文件
        filename = f"{save_dir}/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"   ✅ 数据已保存至: {filename}")
        except Exception as e:
            print(f"   ❌ 保存失败: {e}")
        return {}

    def _keep_memories_alive_node(self, state: State):
        """多轮对话保存记忆"""
        print("💾 [节点] 正在保存记忆...")
        self.memory.save_context(
            {"user_input": state["user_input"]},
            {"output": state["response_obj"].final_response}
        )
        return {}

    def _parse_api_node(self, state: State):
        """分析接口定义"""
        print("\n正在分析文档，提取接口定义...")

        prompt = self.prompt_factory.parse_api_node()
        chain = prompt | self.llm.with_structured_output(ApiDefinition)
        result = chain.invoke({"content": state["context"]})

        print(f"   🛠️ 提取结果: {result.name} -> {result.url}")
        return {"api_definition": result}

    def _generate_casse_node(self, state: State):
        """生成测试用例"""
        print("\n📝 正在设计测试用例...")

        prompt = self.prompt_factory.generate_case_node()
        chain = prompt | self.llm.with_structured_output(TestCase)
        result = chain.invoke({"api_info": state["api_definition"].json()})

        print(f"   🧪 用例: {result.title}")
        return {"test_case": result}

    def _generate_data_node(self, state: State):
        """生成测试数据"""
        print("\n🔢 正在构造测试数据...")

        prompt = self.prompt_factory.generate_data_node()
        chain = prompt | self.llm.with_structured_output(TestData)
        result = chain.invoke({"params": state["api_definition"].parameters,
                               "user_requirement": state["original_input"]
                               })

        print(f"   📦 数据: {result.payload}")
        return {"test_data": result}

    def _generate_assertion_node(self, state: State):
        """生成断言规则"""
        print("\n⚖️ 正在制定断言规则...")

        prompt = self.prompt_factory.generate_assertion_node()

        # 2. 定义工具 (Tool Definition)
        tools = [AssertionRule]

        # 3. 绑定工具
        llm_with_tools = self.llm.bind_tools(tools)

        # 4. 构建 Chain
        chain = prompt | llm_with_tools

        # 5. 调用
        response = chain.invoke({"name": state["api_definition"].name})

        # 6. 提取结果
        if response.tool_calls:
            all_assertions = []
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'AssertionRule':
                    assertion_obj = AssertionRule(**tool_call['args'])
                    all_assertions.append(assertion_obj)
            result = all_assertions[0] if all_assertions else None
        else:
            # 模型没调用工具，返回默认
            print("⚠️ 模型未调用工具，使用默认断言")
            result = AssertionRule(field="status", operator="equals", expected_value="200")

        print(f"   🎯 断言: {result.field} {result.operator} {result.expected_value}")
        return {"assertion": result}

    def _execute_test_node(self, state: State):
        """发送 HTTP 请求"""
        print("\n🚀 正在发起真实 HTTP 请求...")

        api = state["api_definition"]
        data = state["test_data"]
        assertion = state["assertion"]

        # --- 真实请求逻辑 ---
        try:
            print(f"   ⚡ 正在连接: {api.url}")

            # 使用 httpx 发起请求
            with httpx.Client() as client:
                response = client.request(
                    method=api.method,
                    url=api.url,
                    json=data.payload,
                    headers=data.headers,
                    timeout=5.0
                )

                status_code = response.status_code
                response_body = response.text

        except Exception as e:
            # 捕获网络异常（如连接拒绝、超时）
            status_code = 0
            response_body = str(e)
            print(f"   ❌ 请求异常: {e}")

        # --- 断言逻辑 ---
        is_pass = False
        error_msg = None

        try:
            # 尝试解析 JSON
            json_body = json.loads(response_body) if response_body else {}

            # 简单的字段提取（支持一级字段）
            actual_value = json_body.get(assertion.field)

            if assertion.operator == "equals":
                # 如果期望值是字符串数字，尝试转换实际值
                if str(assertion.expected_value).isdigit() and isinstance(actual_value, int):
                    is_pass = (int(assertion.expected_value) == actual_value)
                else:
                    is_pass = (str(assertion.expected_value) == str(actual_value))
            elif assertion.operator == "exists":
                is_pass = (actual_value is not None)
            elif assertion.operator == "contains":
                is_pass = (str(assertion.expected_value) in str(actual_value))

            if not is_pass:
                error_msg = f"断言失败: 期望 {assertion.expected_value}, 实际得到 {actual_value}"

        except json.JSONDecodeError:
            error_msg = "响应不是有效的 JSON"
        except Exception as e:
            error_msg = f"断言执行出错: {str(e)}"

        result = ExecutionResult(
            status_code=status_code,
            response_body=response_body,
            is_success=is_pass,
            error_message=error_msg
        )

        status_text = "✅ 通过" if is_pass else "❌ 失败"
        print(f"   结果: {status_text} (状态码: {status_code})")

        return {"execution_result": result}

    def _generate_report_node(self, state: State):
        """生成报告"""
        print("\n📊 生成最终报告...")

        test_case = state.get("test_case")
        execution_result = state.get("execution_result")

        case_json = test_case.model_dump_json(indent=2) if test_case else "无测试用例信息"
        result_json = execution_result.model_dump_json(indent=2) if execution_result else "无执行结果信息"

        prompt = self.prompt_factory.generate_report_node()
        chain = prompt | self.llm.with_structured_output(TestReport)
        result = chain.invoke({"test_case_info": case_json, "execution_result": result_json})

        print(f"✅ 报告生成完毕: {result.test_title} - {'成功' if result.test_result else '失败'}")
        return print("暂时丢弃，后序专门放个文件存储")