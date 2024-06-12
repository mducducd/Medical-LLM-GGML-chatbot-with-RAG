import asyncio
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from accelerate import Accelerator
from prompts import EXAMPLE_PROMPT, PROMPT, WELCOME_MESSAGE
import ndic
accelerator = Accelerator()

DB_FAISS_PATH = 'vectorstores/db_faiss'

custom_prompt_template = """Use a question from the patient and include Reference information, create a Helpful answer with references as a doctor.
ALWAYS return a more than 200 WORDS in your answer, line by line in detail and include all Reference information

Reference information: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# Loading the model
# def load_llm(model_name="/home/hdd1/duke/llm/ChatDoctor/llama-2-13b-medical-chat", eight_bit=0, device_map="auto"):
#     global model, tokenizer, generator

#     print("Loading "+model_name+"...")

#     if device_map == "zero":
#         device_map = "balanced_low_0"

#     # config
#     gpu_count = torch.cuda.device_count()
#     print('gpu_count', gpu_count)

#     tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)
#     model = transformers.LlamaForCausalLM.from_pretrained(
#         model_name,
#         device_map="balanced_low_0",
#         #device_map="auto",
#         torch_dtype=torch.float16,
#         #max_memory = {0: "14GB", 1: "14GB", 2: "14GB", 3: "14GB",4: "14GB",5: "14GB",6: "14GB",7: "14GB"},
#         #load_in_8bit=eight_bit,
#         #from_tf=True,
#         low_cpu_mem_usage=True,
#         load_in_8bit=False,
#         cache_dir="cache"
#     ).cuda()
#     return  model

# def load_llm():
    
#     model = AutoModelForCausalLM.from_pretrained("/home/hdd1/duke/llm/ChatDoctor/llama-7b-hf", hf=True)
#     tokenizer = AutoTokenizer.from_pretrained("/home/hdd1/duke/llm/ChatDoctor/llama-7b-hf")
#     return model

def load_llm():
    # Load the locally downloaded model here
    config = {'max_new_tokens': 512, 'repetition_penalty': 1.1, 'context_length': 8000, 'temperature':0, 'gpu_layers':50}
    llm = CTransformers(
        model="/home/hdd1/duke/llm/ChatDoctor/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        gpu_layers=50,
        max_new_tokens=512,
        temperature=0.5,
        config=config
    )
    llm, config = accelerator.prepare(llm, config)
    return llm    

# QA Model Function
async def qa_bot():
    print('Loading FAISS')
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cuda'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    
    return qa

# Output function
async def final_result(query):
    qa_result = await qa_bot()
    response = await qa_result({'query': query})
    return response

# chainlit code
@cl.on_chat_start
async def start():
        # Asking user to to upload a PDF to chat with
    files = 'nonone'
    while files is None:
        files = await cl.AskFileMessage(
            content=WELCOME_MESSAGE,
            accept=["application/pdf"],
            max_size_mb=20,
        ).send()
    file = files[0]

    chain = await qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    # answer = res["result"]
    sources = res["source_documents"]

    if sources:
        pass
        # answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    # await cl.Message(content=answer).send()

if __name__ == "__main__":
    asyncio.run(cl.main())
