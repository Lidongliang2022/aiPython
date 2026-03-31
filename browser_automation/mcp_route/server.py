import asyncio
from typing import Any, Optional
from playwright.async_api import async_playwright
from mcp.server.fastmcp import FastMCP

# 初始化 MCP Server
mcp = FastMCP("playwright-server")

# 全局变量来保存 Playwright 实例和页面
browser_context = None
page = None
playwright_instance = None

async def ensure_browser():
    global browser_context, page, playwright_instance
    if playwright_instance is None:
        playwright_instance = await async_playwright().start()
        # 也可以在此处配置持久化 Profile 以便复用本地 Cookie
        # browser_context = await playwright_instance.chromium.launch_persistent_context(
        #     user_data_dir="./mcp_profiles/Default",
        #     headless=False
        # )
        browser = await playwright_instance.chromium.launch(headless=False)
        browser_context = await browser.new_context()
        page = await browser_context.new_page()
    return page

@mcp.tool()
async def open_url(url: str) -> str:
    """打开指定的 URL。"""
    try:
        p = await ensure_browser()
        await p.goto(url)
        return f"已成功打开 {url}"
    except Exception as e:
        return f"打开 {url} 失败: {str(e)}"

@mcp.tool()
async def click_element(selector: str) -> str:
    """在当前页面点击指定的选择器。"""
    try:
        p = await ensure_browser()
        await p.click(selector)
        return f"已点击元素: {selector}"
    except Exception as e:
        return f"点击 {selector} 失败: {str(e)}"

@mcp.tool()
async def input_text(selector: str, text: str) -> str:
    """在当前页面的选择器中输入文本。"""
    try:
        p = await ensure_browser()
        await p.fill(selector, text)
        return f"已在 {selector} 中输入: {text}"
    except Exception as e:
        return f"在 {selector} 中输入失败: {str(e)}"

@mcp.tool()
async def get_page_content() -> str:
    """获取当前页面的 HTML 内容或文本概览。"""
    try:
        p = await ensure_browser()
        # 这里为了演示，只返回文本内容的前 500 个字符
        content = await p.inner_text("body")
        return f"页面内容概览: {content[:500]}..."
    except Exception as e:
        return f"获取内容失败: {str(e)}"

@mcp.tool()
async def take_screenshot() -> str:
    """拍摄当前页面的截图（并保存为 screenshot.png）。"""
    try:
        p = await ensure_browser()
        await p.screenshot(path="mcp_screenshot.png")
        return "截图已保存为 mcp_screenshot.png"
    except Exception as e:
        return f"截图失败: {str(e)}"

if __name__ == "__main__":
    # 使用 stdio 传输协议运行 MCP Server
    mcp.run()
