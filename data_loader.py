from langchain_community.document_loaders import PyPDFLoader



loader = PyPDFLoader(
    file_path="files/paper1.pdf",
)


docs = loader.load()



