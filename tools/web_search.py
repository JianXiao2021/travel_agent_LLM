from langchain_core.tools import tool
from typing import Annotated
import re
import requests
from lxml.html import etree
from duckduckgo_search import DDGS

@tool
def web_search(
    keywords: Annotated[str, "要搜索的关键词，根据你当前的任务目标确定，尽量精确和详细"],
    max_results: Annotated[int, ("最多返回多少条搜索结果。如果返回的搜索结果没有太多有用信息，可以指定返回更多搜索结果")] = 10
) -> list:
    """网络搜索工具。在搜索引擎上搜索关键词，返回指定数目的搜索结果，每个结果包含网页的标题、链接和开头内容。"""
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(
            keywords=keywords,
            region='cn-zh',
            max_results=max_results)]
        return results

# def get_bing_url(keywords):
#     keywords = keywords.strip('\n')
#     bing_url = re.sub(r'^', 'https://cn.bing.com/search?q=', keywords)
#     bing_url = re.sub(r'\s', '+', bing_url)
#     return bing_url

# @tool
# def bing_search(
#     keywords: Annotated[str, "要搜索的关键词，根据你当前的任务目标确定，尽量精确和详细"],
#     page: Annotated[int, ("要返回搜索结果的第几页。由于每页包含25条搜索结果，因此如果page=1，则返回第0-25条结果。"
#                      "如果page=2，则返回第25-50条结果。以此类推")] = 1
# ) -> str:
#     """网络搜索工具。在Bing上搜索关键词，返回指定页的搜索结果，每个结果包含网页的标题和开头内容。"""
#     bing_url = get_bing_url(keywords)
#     headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:81.0) Gecko/20100101 Firefox/81.0',
#                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
#                'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
#                'Accept-Encoding': 'gzip, deflate',
#                }

#     results = []
#     if page <= 1:
#         url = bing_url
#     else:
#         url = bing_url + '&qs=ds&first=' + str((page * 10) - 1) + '&FORM=PERE'
#     # print(url)
    
#     # Get web search results
#     retry_count = 0
#     while retry_count < 3:
#         try:
#             content = requests.get(url=url, timeout=5, headers=headers)
#             content.raise_for_status()  # Raise an HTTPError for bad responses
#             break  # Exit the loop if the request is successful
#         except requests.RequestException as e:
#             retry_count += 1
#             print(f"Request failed (attempt {retry_count}): {e}")
#             if retry_count == 3:
#                 raise e

#     # Concatenate the search results
#     tree = etree.HTML(content.text)
#     li_list = tree.xpath('//ol[@id="b_results"]//li[@class="b_algo"]')
#     for li in li_list:
#         h3 = li.xpath('./h2/a')[0]
#         h3 = h3.xpath('string(.)')
#         p = li.xpath('.//p')[0]
#         p = p.xpath('string(.)')
#         results.append(f"{h3} - {p}")

#     return "\n\n".join(results)

# test the tool
if __name__ == "__main__":
    print(web_search.args_schema.schema())
    a=web_search.invoke({"keywords": "广州 景点"})
    print(a)