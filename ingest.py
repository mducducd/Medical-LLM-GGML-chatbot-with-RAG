from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, MergedDataLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_PATH="data/"
DB_FAISS_PATH="vectorstores_upload/db_faiss"

def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents =loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    ### .txt
    txt_loader = DirectoryLoader(DATA_PATH, glob='*.txt', loader_cls=TextLoader)
    txt_documents =loader.load()
    txt_texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = {'device': 'cuda'})

    loader_all = MergedDataLoader(loaders=[loader, txt_loader])
    
    # db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    db = FAISS.from_documents(texts, embeddings)

    db.save_local(DB_FAISS_PATH)
    print('done!')
if __name__ == "__main__":
    create_vector_db()