from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os

import nest_asyncio
nest_asyncio.apply()

import asyncio
from httpx import AsyncClient, Timeout

async def request_agent(prompt: str, *, system_prompt: str | None = None):
    timeout = Timeout(
        connect=10.0,
        read=300.0,
        write=10.0,
        pool=10.0
    )
    async with AsyncClient(timeout=timeout) as client:
        response = await client.post(
            os.environ.get("LOCAL_URL"),
            headers={
                "Content-Type": "application/json",
                # 这里虽然不需要鉴权，但是需要构造一个鉴权信息
                "Authorization": "Bearer nothing",
            },
            json={
                # 设置本地的模型
                "model": "qwen3:8b",
                "messages": [
                    {"role": "user", "content": 
                     "[角色设定]:\n"
                     f"{ system_prompt }\n"
                     "[用户输入]:\n"
                     f"{ prompt }\n"
                    },
                ],
            },
        )
        return response.json()
    
system_prompt = "你是一个名叫pank的测试专家"
prompt = "你是谁，你能做什么？"

result = asyncio.run(request_agent(prompt, system_prompt= system_prompt))
print(result)