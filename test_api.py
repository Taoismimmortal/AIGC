import httpx

# 定义基础URL
BASE_URL = "http://127.0.0.1:8000"  # 确保FastAPI服务运行在该地址

def test_root():
    """测试根路径"""
    response = httpx.get(f"{BASE_URL}/")
    if response.status_code == 200:
        print("Root Path Response:", response.json())
    else:
        print("Root Path Error:", response.status_code, response.text)

def test_get_response():
    """测试普通响应路由"""
    payload = {
        "session_id": "test_session",
        "input": "你好，小海，gdou什么时候放假？"
    }
    response = httpx.post(f"{BASE_URL}/get_response/", json=payload)
    if response.status_code == 200:
        print("Get Response:", response.json())
    else:
        print("Get Response Error:", response.status_code, response.text)

def test_stream_response():
    """测试流式响应路由"""
    payload = {
        "session_id": "test_session",
        "input": "你好，今天有什么新闻？"
    }
    with httpx.stream("POST", f"{BASE_URL}/stream_response/", json=payload) as response:
        if response.status_code == 200:
            print("Stream Response:")
            for chunk in response.iter_text():
                print(chunk, end="")
        else:
            print("Stream Response Error:", response.status_code, response.text)

if __name__ == "__main__":

    test_root()


    test_get_response()


    test_stream_response()
