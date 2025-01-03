from pydantic import BaseModel
from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools import LocalSearch, WebSearch, WebVisit
from typing import Optional
from dotenv import load_dotenv
import os
#############
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_redis import RedisChatMessageHistory

# 加载环境变量
load_dotenv()

# 定义用户输入的数据模型
class UserInput(BaseModel):
    session_id: str
    input: str
    output: Optional[str]
# 定义 AiGuide 类
class AiGuide:
    # 代理执行器
    agent_executor = None
    # 系统提示信息，包含对校园助手的工作规则说明
    sys_prompt = (
    "You are a helpful guide for the Computer Science and Engineering college of GDOU called '小海'. "
    "Your task is to answer questions about the campus from students. You should always follow the following rules to work:\n"
    "1. Analyze the user’s question and extract one key word that is relevant to the campus or the topic the user is asking about.\n"
    "2. Based on the extracted keyword, decide if you need to use a tool to find more information. If so, proceed with one of the following options:\n"
    "    a. If the keyword requires a search engine, use the WebSearch tool to search for the keyword.\n"
    "    b. If local documents or resources might contain relevant information, use the appropriate tool to retrieve it.\n"
    "3. If you used the WebSearch tool, you may use another tool to retrieve the content of one web page that could be helpful, but only use it once or twice, and optionally provide the URL.\n"
    "4. Summarize the gathered information clearly and concisely, answer the user’s question, and provide the source of the information at the end of your answer.\n"
    "5. If no relevant information is found, ask the user for more details or offer an apology for not being able to find an answer.\n"
    "6. At the end of your answer, welcome the user to GDOU and invite them to ask more questions about the campus.\n"
    "final answer: "
)

    # 带有聊天历史的代理
    agent_with_chat_history = None

    # 标记是否使用流
    stream: bool = False

    def __init__(self, streams: bool = False):
        # Redis 配置
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_namespace = "aiguide:chat"  # 命名空间，避免不同项目冲突

        self.stream = streams
        # 创建 ChatOpenAI 实例，设置模型、API 密钥、基础 URL 和是否使用流
        model = ChatOpenAI(
            model_name=os.getenv("ZHIPU_MODEL"),
            openai_api_key=os.getenv("ZHIPU_API_KEY"),
            openai_api_base=os.getenv("ZHIPU_BASE_URL")
        )
        # 从 hub 中拉取提示信息，这里使用了另一种方式创建提示模板
        # prompt = hub.pull("hwchase17/react-chat")
        prompt = ChatPromptTemplate.from_messages(
            [
                # 系统提示
                ("system", self.sys_prompt),
                # 聊天历史占位符
                MessagesPlaceholder(variable_name="chat_history"),

                # 用户输入占位符
                ("user", "{input}"),
                # 代理暂存区占位符
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        # 定义工具列表
        tools = [WebSearch()]

        # 创建工具调用代理
        agent = create_tool_calling_agent(model, tools, prompt)
        # 也可以使用另一种方式创建代理
        # agent = create_react_agent(model, tools, prompt)
        # 创建代理执行器
        agent_executor = AgentExecutor(agent=agent, tools=tools)
        # 创建带有消息历史的可运行对象
        # 历史运行器
        self.agent_with_chat_history = RunnableWithMessageHistory(
            agent_executor,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    # 获取会话历史的方法
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        return RedisChatMessageHistory(
            session_id=session_id,
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),

        )

    # 带有历史的调用方法
    def invoke_with_history(self, user_input: UserInput, stream=False):
        print(f"User Input: {str(user_input)}")

        return self.agent_with_chat_history.invoke(
            {
                "input": user_input.input
            },
            config={"configurable": {"session_id": user_input.session_id}}
        )["output"]


if __name__ == "__main__":
    # 创建 AiGuide 实例并调用带有历史的调用方法，传入用户输入信息
    aiguide = AiGuide(streams=True)

    # 调用带有历史的调用方法，传入用户输入信息
    response1 = aiguide.invoke_with_history(
        UserInput(session_id="test", input="我是K,你好", output="")
    )
    print("AI Response 1:", response1)

    response2 = aiguide.invoke_with_history(
        UserInput(session_id="test", input="广东海洋大学有什么新闻", output="")
    )
    print("AI Response 2:", response2)
