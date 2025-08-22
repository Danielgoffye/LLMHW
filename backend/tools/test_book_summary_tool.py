import sys
import os

sys.path.append(os.path.abspath("backend"))
from tools.book_summary_tool import get_summary_by_title

title = "Harry Potter and the Philosopher's Stone"
summary = get_summary_by_title(title)

if summary:
    print(f"Summary for '{title}':\n{summary}")
else:
    print(f"No summary found for '{title}'")
