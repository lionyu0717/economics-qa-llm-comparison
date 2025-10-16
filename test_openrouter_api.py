# This file will be populated with OpenRouter test code. 

import os
from openai import OpenAI
from dotenv import load_dotenv

def test_openrouter_connectivity():
    """
    Tests connectivity to OpenRouter and API key validity.
    """
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    site_url = os.getenv("YOUR_SITE_URL") # Optional
    site_name = os.getenv("YOUR_SITE_NAME") # Optional

    if not api_key:
        print("错误：OPENROUTER_API_KEY 未在环境变量中找到。")
        print("请确保：")
        print("1. 你的 .env 文件在当前工作目录。")
        print("2. .env 文件中包含 OPENROUTER_API_KEY=your_api_key 这一行。")
        return

    print(f"已找到 OpenRouter API 密钥 (末尾四位): ...{api_key[-4:]}")
    if site_url:
        print(f"使用 HTTP-Referer: {site_url}")
    if site_name:
        print(f"使用 X-Title: {site_name}")

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        headers = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_name:
            headers["X-Title"] = site_name

        print("\n正在尝试使用 OpenRouter (google/gemini-2.0-flash-001) 进行一次简单的API调用...")
        
        completion = client.chat.completions.create(
            model="google/gemini-2.0-flash-001", # Or use a cheaper/free model for testing like "mistralai/mistral-7b-instruct-v0.1"
            messages=[
                {
                    "role": "user",
                    "content": "你好！请用中文简单介绍一下你自己。"
                }
            ],
            temperature=0.1,
            max_tokens=150,
            extra_headers=headers if headers else None
        )
        
        print("\nAPI 调用成功！")
        print("模型的回复:")
        print(completion.choices[0].message.content)
        print("\n如果看到有意义的回复，说明您的 OpenRouter API 密钥配置正确且有效！")

    except Exception as e:
        print(f"\nAPI 调用过程中发生错误: {e}")
        print("\n请仔细检查以下几点：")
        print("1. OpenRouter API 密钥是否完全正确且有效。")
        print("2. 你的 OpenRouter 账户是否有足够的额度。")
        print("3. 网络连接是否正常，能否访问 https://openrouter.ai/")
        print(f"   错误类型: {type(e).__name__}")
        if hasattr(e, 'response') and e.response is not None and hasattr(e.response, 'text'):
            print(f"   API 返回的详细错误: {e.response.text}")
        elif hasattr(e, 'message'):
             print(f"   错误消息: {e.message}")

if __name__ == "__main__":
    print("开始测试 OpenRouter API 密钥和调用...")
    print("======================================")
    test_openrouter_connectivity()
    print("======================================")
    print("测试结束。") 