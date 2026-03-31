# 路线二：基于 MCP 的浏览器自动化 (Playwright Server)

这是基于 **Model Context Protocol (MCP)** 的架构实现。将浏览器控制逻辑（Server）与任务决策逻辑（Client/LLM）完全解耦。

## 架构说明

1. **MCP Server (`server.py`)**: 
   - 使用 `mcp` SDK 构建。
   - 封装 Playwright 能力（打开 URL、点击、输入、截图）。
   - 通过标准输入输出 (stdio) 与客户端通信。
2. **MCP Client (`client.py`)**:
   - 使用 `langchain-openai` 和 `mcp` SDK 构建。
   - 动态发现并绑定服务器暴露的工具。
   - 实现大模型（GPT-4o/Claude 等）与物理浏览器的循环交互。

## 快速开始

1. **环境准备**:
   ```bash
   pip install -r browser_automation/mcp_route/requirements.txt
   playwright install chromium
   ```

2. **配置环境变量**:
   在根目录或 `browser_automation/mcp_route/` 创建 `.env` 并填写：
   ```env
   OPENAI_API_KEY=your_key
   OPENAI_API_BASE=https://api.openai.com/v1
   ```

3. **运行演示客户端**:
   ```bash
   python browser_automation/mcp_route/client.py
   ```

## 注意事项

- **磁盘空间问题**: 
  如果之前遇到 `No space left on device` 错误，是因为之前的脚本在 `/tmp` 或 `/var/folders` 下生成了大量 Chrome Profile 副本。
  您可以运行以下命令清理临时空间：
  ```bash
  rm -rf /var/folders/s4/4prrmbrx0wgctb63jtqcfy3c0000gn/T/browser-use-user-data-dir-*
  ```
- **iframe 支持**: 
  在 MCP Server 中，您可以更精细地控制 Playwright，例如使用 `page.frames` 来显式切换 iframe 逻辑。

## 扩展建议
- 可以在 `server.py` 中使用 `launch_persistent_context` 并指定固定目录，以复用本地浏览器的 Cookie。
- 可以在 `client.py` 中集成更复杂的 Agent 框架（如 LangGraph）以处理更复杂的浏览器自动化流程。
