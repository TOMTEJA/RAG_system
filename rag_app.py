import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load the embedding model
f = open('KEYS/api_key.txt')
key = f.read()
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=key,model="models/embedding-001")
db_connection = Chroma(persist_directory="RAG_embeddings",embedding_function=embedding_model)

# Define retrieval function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) 
# Converting CHROMA db_connection to Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

# Define chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a Helpful AI Bot. 
    Your task is to provide assistance based on the context given by the user. 
    Make sure your answers are relevant and helpful."""),
    HumanMessagePromptTemplate.from_template("Answer the question based on the given context.\nContext:\n{context}\nQuestion:\n{question}\nAnswer:")
])
# Initialize chat model
chat_model = ChatGoogleGenerativeAI(google_api_key=key,model="gemini-1.5-pro-latest")
output_parser = StrOutputParser()
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

# Streamlit UI
st.title("🤖AI RAG Q&A SYSTEM ON 'LEAVE NO CONTEXT BEHIND' Paper")
st.subheader("Get answers based on this paper")

question = st.text_input("", placeholder="Enter your question here...")
if st.button("Generate Answer"):
    if question:
        response = rag_chain.invoke(question)
        st.write("📝📣 Answer:")
        st.write(response)
    else:
        st.warning("📑💡Please enter a question.")
