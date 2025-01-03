from typing import Optional, Type
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class WebSearchInput(BaseModel):
    keyword: str = Field(description="One word to search for")

class WebSearch(BaseTool):
    name: str = "Web Information Search"
    description: str = "Useful for when you need to search for information about campus in GDOU's official website."
    args_schema: Type[BaseModel] = WebSearchInput
    return_direct: bool = False
    query_header: str = "site:gdou.edu.cn"

    def _run(
        self, keyword: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> list:
        """Use the tool."""
        if not keyword.startswith(self.query_header):
            keyword = self.query_header + ' ' + keyword
        print(f"Final keyword: {keyword}")
        with DDGS() as duck:
            web_query = [r for r in duck.text(keyword, region='cn-zh', max_results=4)]
            filtered_results = [res for res in web_query if "gdou.edu.cn" in res.get('href', '')]
            print(f"Filtered results: {filtered_results}")
            return filtered_results

    async def _arun(
        self,
        keyword: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> list:
        """Use the tool asynchronously."""
        return self._run(keyword, run_manager=run_manager.get_sync())

if __name__ == "__main__":
    tool = WebSearch()
    results = tool._run("广东海洋大学校长")
    print(results)
