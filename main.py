# Author: ZK-Jackie
# Time: 2024/08/09
# Title: CampusQA
# 导入所需的模块和函数
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse

from api.chat import api_chat
from auth import verify_host, verify_token
import uvicorn

# 创建 FastAPI 应用实例，并添加依赖项
app = FastAPI(dependencies=[Depends(verify_host), Depends(verify_token)])

# 添加 CORS 中间件，允许跨域请求
app.add_middleware(
        CORSMiddleware,
        # 允许的源
        allow_origins=["*"],
        # 允许携带凭证
        allow_credentials=True,
        # 允许的请求方法
        allow_methods=["*"],
        # 允许的请求头
        allow_headers=["*"],
)

# 定义一个 HTTP 中间件函数，用于拦截所有请求
@app.middleware("http")
async def intercept_all_requests(request: Request, call_next):
    # 在这里处理所有请求
    try:
        # 验证请求的主机
        verify_host(request)
        # 验证请求的令牌
        verify_token(request, request.headers.get("Token"))
    except HTTPException as e:
        # 如果验证失败，返回相应的 HTTP 异常响应
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    # 调用下一个处理函数
    response = await call_next(request)
    return response

# 包含聊天 API 的路由
app.include_router(api_chat)
# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")




if __name__ == "__main__":
    uvicorn.run("main:app",
                host="127.0.0.1",
                port=54823,
                reload=True)
