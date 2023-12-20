import os
os.environ["OPENAI_API_KEY"] = "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"

# import dotenv

# dotenv.load_dotenv()
import streamlit as st
#from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Pinecone
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from operator import itemgetter
from langchain_core.runnables import RunnableParallel
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.docstore.document import Document
from streamlit_chat import message
import tempfile
from PyPDF2 import PdfReader
import pinecone
#import bs4


pinecone.init(api_key='d49011f7-9f67-4f17-a092-38a22e0cbe83', environment='gcp-starter')
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
#index = pinecone.Index('chatbot')
uploaded_file = st.sidebar.file_uploader("Upload documents", type="pdf")

    #i = 1
    #for page in reader.pages:
        #docs.append(Document(page_content=page.extract_text(), metadata={'page':i}))
        #i += 1
if uploaded_file is None:
  st.info("""Upload files to analyse""")
elif uploaded_file  is not None:
    docs = []
    reader = PdfReader(uploaded_file)
    i = 1
    for page in reader.pages:
        docs.append(Document(page_content=page.extract_text(), metadata={'page':i}))
        i += 1
    #with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        #tmp_file.write(uploaded_file.getvalue())
        #tmp_file_path = tmp_file.name

    #loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    #reader = PyPDFLoader(file_path=tmp_file_path)  
      
    #data = reader.load_and_split()
    
#st.header("MEDBOT")
#st.write("---")
#uploaded_files = st.file_uploader("Upload documents",accept_multiple_files=True, type=["pdf"])
#uploaded_files = st.sidebar.file_uploader("upload", type="pdf")
#from langchain.document_loaders import WebBaseLoader

#loader = WebBaseLoader(
    #web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    #bs_kwargs={
        #"parse_only": bs4.SoupStrainer(
           # class_=("post-content", "post-title", "post-header")
       # )
   # },
#)
#docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs)

vectorstore = Pinecone.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(), index_name='chatbot')

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
  
#retrieved_docs = retriever.get_relevant_documents("what is a sigmoid colon?")
template = """You are a virtual clinic coordinator for Advanced Surgeons. 
Greet people politely who say hi or hello. When you know the answer, give the source document and page number you have taken the information from.
Also give a link to the source document.
DO NOT PERFORM AN INTERNET SEARCH. DO NOT ACCESS THE TRAINING DATA. 
If you do not know the answer to their question or have no information, 
do not guess but say this exactly: " I don't know the answer to the question. 
Click on the Call Us link to be connected to a coordinator " followed by a clickable link for tel:9142827802 with caption "Call us"
{context}
Question: {question}
Helpful Answer:"""

rag_prompt_custom = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | rag_prompt_custom | llm| StrOutputParser())


for chunk in rag_chain.stream("what is sigmoid colon?"):
    print(chunk, end="", flush=True)


rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | rag_prompt_custom
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"documents": retriever, "question": RunnablePassthrough()}
) | {
    "documents": lambda input: [doc.metadata for doc in input["documents"]],
    "answer": rag_chain_from_docs,
}


condense_q_system_prompt = """Given a chat history and the latest user question \
which might reference the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

condense_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

condense_q_chain = condense_q_prompt | llm | StrOutputParser()

qa_system_prompt = """You are a virtual clinic coordinator for Advanced Surgeons. 
Greet people politely who say hi or hello. When you know the answer, give the source document and page number you have taken the information from.
Also give a link to the source document.
DO NOT PERFORM AN INTERNET SEARCH. DO NOT ACCESS THE TRAINING DATA. 
If you do not know the answer to their question or have no information, 
do not guess but say this exactly: " I don't know the answer to the question. 
Click on the Call Us link to be connected to a coordinator " followed by a clickable link for tel:9142827802 with caption "Call us"
{context}
Question: {question}
Helpful Answer:"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

def condense_question(input: dict):
    if input.get("chat_history"):
        return condense_q_chain
    else:
        return input["question"]

rag_chain = (
    RunnablePassthrough.assign(context=condense_question | retriever | format_docs)
    | qa_prompt
    | llm
)

chat_history = []

def add_character(text):
    character = "!"
    return text + character

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello"]

if 'past' not in st.session_state:
    st.session_state['past'] = []

response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Ask me questions", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = rag_chain_with_source.invoke({"question": user_input, "chat_history": chat_history})

        st.session_state['history'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["history"][i], is_user=True, key=str(i) + '_user')
        
            #st.write(f"Generated: {st.session_state['generated'][i]}")
            #st.write(f"User: {st.session_state['history'][i]}")
            

