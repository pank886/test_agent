from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 定义 System Prompt 的模板内容
SYSTEM_TEMPLATE = (
    "你是一个资深测试工程师。"
    "请根据以下的【参考资料】和【对话历史】进行回答。"
    "如果【参考资料】里有答案，优先依据资料回答；如果没有，请依据你的专业知识。"
    "\n\n【参考资料】:\n{context}"
)

class PromptFactory:

    def get_prompt_template(self) -> ChatPromptTemplate:
        """
        返回配置好的 Prompt 模板对象
        """
        return ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_input}")
        ])

    def parse_api_node(self) -> ChatPromptTemplate:
        """
        分析接口
        """
        return ChatPromptTemplate.from_messages([
            ("system",
             "你是一个API架构师。请从文档内容中提取结构化的接口定义。"
             "注意：严格根据文档内容生成。"),
            ("human", "文档内容:\n{content}")
        ])

    def generate_case_node(self) -> ChatPromptTemplate:
        """
        生成测试用例
        """
        return ChatPromptTemplate.from_messages([
            ("system", "你是一个测试专家。根据接口定义设计一个正向测试用例。"),
            ("human", "接口信息:\n{api_info}\n\n请生成测试用例。")
        ])

    def generate_data_node(self) -> ChatPromptTemplate:
        """
        生成测试数据
        """
        return ChatPromptTemplate.from_messages([
            ("system", "你是一个数据构造助手。根据接口参数定义以及提供的相关数据，生成一套合法的测试数据。"),
            ("human", "参数定义:\n{params}\n相关数据:\n{user_requirement}\n\n请生成具体的 payload。")
        ])

    def generate_assertion_node(self) -> ChatPromptTemplate:
        """
        生成断言
        """
        return ChatPromptTemplate.from_messages([
            ("system",
             "你是一个QA专家。针对这个接口，定义一个核心的成功断言（例如检查 HTTP 状态码或返回体中的 code 字段）。"),
            ("human", "接口: {name}\n\n请定义断言规则。")
        ])

    def generate_report_node(self) -> ChatPromptTemplate:
        """
        生成测试报告
        """
        return ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的测试报告生成助手。请根据提供的【测试用例信息】和【执行结果】，生成一份简洁的测试报告。"),
            ("human",
             "### 测试用例信息:\n{test_case_info}\n\n"
             "### 执行结果详情:\n{execution_result}\n\n"
             "请根据以上信息生成测试报告。")
        ])