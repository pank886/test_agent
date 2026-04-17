from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class ProperResponse(BaseModel):
    """
    定义了 LLM 输出数据的结构。
    """
    proper_thinking: List[str] = Field(description="针对如何回复这个问题的思考")
    final_response: str = Field(description="整理思考后的最终回复")
    worth_to_remember: bool = Field(description="从测试经验提升角度判断是否值得记忆")

class ApiDefinition(BaseModel):
    name: str = Field(description="接口名称")
    url: str = Field(description="接口完整路径，如 http://localhost:8000/api/login")
    method: str = Field(description="HTTP方法: GET, POST, PUT, DELETE")
    description: str = Field(description="接口功能描述")
    parameters: Dict[str, Any] = Field(description="请求参数结构示例")

class TestCase(BaseModel):
    title: str = Field(description="测试用例标题")
    description: str = Field(description="测试目的描述")
    pre_condition: str = Field(description="前置条件")

class TestData(BaseModel):
    payload: Dict[str, Any] = Field(description="具体的请求体 JSON 数据")
    headers: Dict[str, str] = Field(default_factory=dict, description="请求头")

class AssertionRule(BaseModel):
    field: str = Field(description="需要校验的返回字段，如 'code' 或 'data.status'")
    operator: str = Field(description="比较操作符: 'equals', 'contains', 'exists'")
    expected_value: Any = Field(description="期望的值")

class ExecutionResult(BaseModel):
    status_code: int
    response_body: str
    is_success: bool
    error_message: Optional[str] = None

class TestReport(BaseModel):
    test_title: str = Field(description="执行的测试用例标题")
    test_description: str = Field(description="测试用例目的描述")
    test_result: bool = Field(description="是否测试通过")