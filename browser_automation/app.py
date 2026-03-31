import os
import sys
import asyncio
import streamlit as st
from browser_use import Agent, Browser, BrowserProfile, ChatOpenAI
# from langchain_openai import ChatOpenAI  # 不再使用 langchain 的 ChatOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

st.set_page_config(page_title="AI Browser Agent", layout="wide")

st.title("🌐 AI 浏览器操作助手 (browser-use)")

# 侧边栏配置
with st.sidebar:
    st.header("配置参数")
    
    api_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    base_url = st.text_input("API Base URL", value=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
    model_name = st.selectbox("选择模型", ["doubao-seed-2.0-pro"], index=0)
    
    st.divider()
    
    # 浏览器 Profile 配置
    default_profile_path = os.path.expanduser("~/Library/Application Support/Google/Chrome")
    profile_path = st.text_input("Chrome User Data Dir", value=default_profile_path)
    profile_name = st.text_input("Profile Directory (e.g. Default, Profile 1)", value="Default")
    
    headless = st.checkbox("无头模式 (Headless)", value=False)

# 主界面
task_input = st.text_area("输入任务目标", placeholder="例如：去 GitHub 搜索 'browser-use'，并告诉我它有多少个 star。", height=100)

if st.button("开始执行", type="primary") and task_input:
    if not api_key:
        st.error("请先在侧边栏配置 API Key")
    else:
        # 准备显示日志
        log_container = st.container()
        log_container.subheader("执行日志")
        
        # 实时打印日志的占位符
        status_placeholder = st.empty()
        
        async def main():
            try:
                # 配置浏览器
                # 在 browser-use 0.12.5 中，使用 BrowserProfile 代替 BrowserConfig
                profile = BrowserProfile(
                    headless=headless,
                    user_data_dir=profile_path,
                    profile_directory=profile_name
                )
                # Browser (BrowserSession) 接收 browser_profile 参数
                browser = Browser(browser_profile=profile)
                
                # 初始化 LLM
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    base_url=base_url
                )
                
                # 初始化 Agent
                agent = Agent(
                    task=task_input,
                    llm=llm,
                    browser=browser,
                )
                
                status_placeholder.info("🚀 Agent 正在启动...")
                
                # 执行任务
                # browser-use 0.12.5 的 Agent.run() 返回 AgentHistoryList
                history = await agent.run()
                
                status_placeholder.success("✅ 任务执行完成！")
                
                # 显示执行历史/结果
                st.subheader("执行结果总结")
                # history.history 是历史步骤列表
                for i, item in enumerate(history.history):
                    if item.model_output:
                        thought = item.model_output.thinking or "无思考过程"
                        # 截断 thought 用于标题
                        title_thought = thought[:50] + "..." if len(thought) > 50 else thought
                        with st.expander(f"步骤 {i+1}: {title_thought}"):
                            st.write("**思考:**", thought)
                            if item.model_output.action:
                                st.write("**动作:**", item.model_output.action)
                            
                            # 显示每一步的结果
                            if item.result:
                                for res in item.result:
                                    if res.extracted_content:
                                        st.write("**提取内容:**", res.extracted_content)
                                    if res.error:
                                        st.error(f"**错误:** {res.error}")
                                    if res.is_done:
                                        st.success(f"**完成状态:** {'成功' if res.success else '失败'}")
                
                if history.final_result():
                    st.info(f"**最终结果:** {history.final_result()}")
                elif history.history and history.history[-1].result:
                    st.info(f"**步骤结果:** {history.history[-1].result}")
                    
            except Exception as e:
                st.error(f"❌ 执行出错: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                # 确保浏览器关闭 (如果不是持久化模式或者需要手动管理)
                pass

        # 运行异步任务
        asyncio.run(main())
else:
    st.info("请在上方输入任务并点击“开始执行”")

# 添加说明
with st.expander("使用说明"):
    st.markdown("""
    1. **环境准备**: 确保已安装 `browser-use`, `langchain-openai`, `playwright`, `streamlit`。
    2. **API Key**: 建议使用环境变量 `OPENAI_API_KEY` 或在侧边栏手动输入。
    3. **持久化 Profile**: 
       - 默认路径为 macOS 的 Chrome 路径。
       - **注意**: 执行时请关闭已打开的该 Profile 的 Chrome 浏览器，否则 Playwright 可能会因为文件被占用而无法启动。
    4. **日志查看**: 执行过程中，Agent 的思考过程和动作将会在页面下方实时更新。
    """)
