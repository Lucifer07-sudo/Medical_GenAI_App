from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings



#extract data from PDF

def load_pdf_file(data):
    '''Load file from directory, data=directory path,
    glob=file type, loader_cls=loader to use
    Usage: load_pdf_file(data=Data/)
    '''
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    return documents


#Split data into text chunks
def text_split(extracted_data):
    '''split documents into chunks. Example: text_split(data)'''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


#Download embedding model from HuggingFaceEndpoint
def download_huggingface_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings