# flake8: noqa
from langchain.prompts import PromptTemplate

# template = """Given the following extracted parts of a long document and a question from the patient, create a final answer with references as a doctor("SOURCES").
# If you don't know the answer, just say that you don't know. Don't try to make up an answer.
# ALWAYS return a "SOURCES" field in your answer and never recommend to meet a doctor,  consult a doctor or something similar; with the format "SOURCES: <source1>, <source2>, <source3>, ...".

# QUESTION: {question}
# =========
# {summaries}
# =========
# FINAL ANSWER:"""

template = """Use a question from the patient and include Reference information, create a final answer with references as a doctor.
ALWAYS return a more than 200 WORDS in your answer, line by line in detail and include all Reference information

QUESTION: {question}
=========
Reference information: {summaries}
=========
FINAL ANSWER:"""


PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "question"]
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)

WELCOME_MESSAGE = """\
Welcome to Medical Chatbot Consultant!
To get started:
1. Upload a PDF or text file
2. Ask any question for medical advices!
"""
