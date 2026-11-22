"""
Prompt templates used by router engines to pick the single best choice (tool or index)
among multiple choices based on a query.
"""

def get_single_select_prompt(language: str):
    """
    Returns the appropriate selection prompt template depending on the selected language ('vi' or 'eng').
    Used by LLMSingleSelector in RouterRetriever to determine which retrieval tool to execute.
    """
    if language == "vi":
        return single_select_prompt_vi
    return single_select_prompt_en


# English single selection prompt for RouterRetriever
single_select_prompt_en = (
    "Some choices are given below. It is provided in a numbered list "
    "(1 to {num_choices}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return "
    "ONE AND ONLY ONE choice that is most relevant to the query: '{query_str}'\n"
)

# Vietnamese single selection prompt for RouterRetriever
single_select_prompt_vi = (
    "Dưới đây là một số lựa chọn được đưa ra, được cung cấp trong một danh sách có số thứ tự "
    "(từ 1 đến {num_choices}), "
    "trong đó mỗi mục trong danh sách tương ứng với một tóm tắt.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Chỉ sử dụng các lựa chọn ở trên và không dùng kiến thức trước đó, hãy chọn "
    "1 và chỉ 1 lựa chọn mà liên quan nhất đến câu truy vấn: '{query_str}'\n"
)

