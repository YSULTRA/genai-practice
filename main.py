from dotenv import load_dotenv

load_dotenv()

from data_loader import docs
from text_splitter import splitter
from embedding import embedding_model
from vector_store import vectorStore
from langchain_classic.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model


split_results = splitter.split_documents(documents=docs)
#chunking done


#embeddings
embedding_model = embedding_model


#vector Store
vectorStore.add_documents(
    documents=split_results
)



retriver = vectorStore.as_retriever(
    search_type="mmr", search_kwargs={"k": 6,"fetch_k": 10, "lambda_mult": 0.25}
)


llm = init_chat_model(
    model = "google_genai:gemini-3.1-flash-lite-preview"
)


prompt = ChatPromptTemplate(
    [
        ("system","You are helpful ai assistnat which answer query answer using context given if you dont have answer just say i cannot answer this question"),
        ("human","you context is {context} and your query is {query}")
    ]
)


while True:

    query = input("You input: ")

    if query == "0":
        print("thank you ")
        break
    
    
    res = retriver.invoke(input=query)
    context = ""

    for doc in res:
        context += doc.page_content
    
  
    
    finalPrompt = prompt.invoke(
        {
            "context":context,
            "query" : query
        }
    )


    output = llm.invoke(input = finalPrompt)

    print("Ai :")
    print(output.content[0]["text"])







