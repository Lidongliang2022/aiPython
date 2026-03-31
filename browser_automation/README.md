# Browser Automation Project (browser-use)

这是一个基于 `browser-use` 和 `Streamlit` 的浏览器自动化探索项目。

## 快速开始

1. **进入目录**:
   ```bash
   cd browser_automation
   ```

2. **安装依赖** (如果尚未安装):
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

3. **配置环境**:
   - 复制 `.env.example` 为 `.env`
   - 填写你的 `OPENAI_API_KEY`

4. **运行应用**:
   ```bash
   streamlit run app.py
   ```

## 功能特点

- **原生脚本流**: 基于 `browser-use` 核心库。
- **持久化会话**: 支持连接到本地 Chrome Profile，复用登录状态。
- **实时监控**: 通过 Streamlit 界面实时查看 Agent 的思考过程和执行步骤。

## 注意事项

- 在运行 Agent 时，如果指定了本地 Chrome Profile，请确保该 Profile 的 Chrome 浏览器已完全关闭，否则可能会导致启动失败。
- 推荐使用 GPT-4o 或 Claude 3.5 Sonnet 等支持 Function Calling 的模型以获得最佳效果。
