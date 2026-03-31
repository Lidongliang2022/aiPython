import asyncio
import os
import json
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv

# 加载环境变量 (确保已有 OPENAI_API_KEY)
load_dotenv()

async def run_mcp_agent():
    # 1. 配置并启动本地 MCP Server (stdio 模式)
    server_params = StdioServerParameters(
        command="python",
        args=["browser_automation/mcp_route/server.py"]
    )
    
    # 2. 与 MCP Server 建立通信 Session
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化并列出服务器提供的工具
            await session.initialize()
            mcp_tools = await session.list_tools()
            print(f"✅ 成功连接到 MCP Server，发现工具: {[tool.name for tool in mcp_tools]}")

            # 3. 将 MCP 工具转换为 LLM 可理解的工具定义 (OpenAI 格式)
            openai_tools = []
            for tool in mcp_tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })

            # 4. 初始化 LLM 并绑定工具
            llm = ChatOpenAI(
                model="gpt-4o",
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            ).bind_tools(openai_tools)

            # 5. 定义任务并让 LLM 决策
            # 示例任务：打开百度并搜索 'MCP Protocol'
            messages = [HumanMessage(content="打开百度首页 https://www.baidu.com，并搜索 'MCP Protocol'。")]
            
            print(f"🚀 发送任务给 LLM: {messages[0].content}")
            ai_msg = await llm.ainvoke(messages)
            messages.append(ai_msg)

            # 6. 如果 LLM 决定调用工具，则通过 MCP Session 执行工具请求并回传结果
            while ai_msg.tool_calls:
                for tool_call in ai_msg.tool_calls:
                    print(f"🛠️  LLM 请求调用工具: {tool_call['name']}({tool_call['args']})")
                    
                    # 通过 MCP Session 调用服务器端工具
                    result = await session.call_tool(tool_call['name'], tool_call['args'])
                    
                    # 将结果传回给 LLM
                    messages.append(ToolMessage(
                        tool_call_id=tool_call['id'],
                        content=str(result.content)
                    ))
                
                # 再次让 LLM 分析当前状态并给出下一个动作
                print("🔄 发送工具执行结果给 LLM...")
                ai_msg = await llm.ainvoke(messages)
                messages.append(ai_msg)
            
            print(f"🏁 任务最终回复: {ai_msg.content}")

if __name__ == "__main__":
    asyncio.run(run_mcp_agent())
