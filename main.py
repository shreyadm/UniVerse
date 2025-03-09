import chainlit as cl
from langchain_community.embeddings import OpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
from passlib.hash import bcrypt
import numpy as np

USER_DB = {
    "Shreya": bcrypt.hash("ZaQ1@wSx"),  # Hash the password for security
    "user": bcrypt.hash("qwerty123")
}

embeddings = OpenAIEmbeddings()
embedding_model_for_pdf = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(temperature=0, model_name='gpt-4o', streaming=True)

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Validates username and password"""
    if username in USER_DB and bcrypt.verify(password, USER_DB[username]):
        return cl.User(
            identifier=username,
            metadata={"role": "admin" if username == "Shreya" else "user", "provider": "credentials"}
        )
    return None  # Authentication failed

@cl.on_chat_start
async def on_chat_start():
    user = cl.user_session.get("user")
    await cl.Message(content=f"Hi {user.identifier}, how may I help you today?").send()

async def retrieve_answer(vector_db_path, embedding_model, query, llm, category):
    """Retrieve answer and similarity score from a vector database."""
    similarity_scores = []
    vector_store = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    # Retrieve documents with similarity scores
    retrieved_docs_with_scores = vector_store.similarity_search_with_score(query, k=3)
    for doc, score in retrieved_docs_with_scores:
        similarity_scores.append(score)
    if category == "pdf_data":
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        response = await chain.acall(query)
        answer = response.get("result", "No answer found.")
    else:
        chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)
        response = await chain.acall(query)
        answer = response.get("answer", "No answer found.")
    
    # Get the max similarity score among the retrieved docs
    max_similarity_score = max(similarity_scores) if similarity_scores else 0
    return answer, max_similarity_score

@cl.on_message
async def on_message(message: cl.Message):
    query = message.content

    # Send a loading message
    msg = cl.Message(content="🔄 Thinking...")
    await msg.send()  # Send the initial loading message
    
    # Define paths for both vector databases
    pdf_vector_db_path = r"C:\Shreya\Final Year Project\chainlit-chatbot\Faiss_vectorDB_PDF_data"
    web_vector_db_path = r"C:\Shreya\Final Year Project\chainlit-chatbot\Faiss_vectorDB_website_data"
    
    # Retrieve answers and similarity scores from both databases
    pdf_answer, pdf_score = await retrieve_answer(pdf_vector_db_path, embedding_model_for_pdf, query, llm, "pdf_data")
    web_answer, web_score = await retrieve_answer(web_vector_db_path, embeddings, query, llm, "website_data")
    
    # Select the answer with the higher similarity score
    if pdf_score > web_score:
        final_answer = pdf_answer
        similarity_score_str = f"\n\n\nBased on PDF data with similarity score: {pdf_score:.2f}\nWeb score: {web_score:.2f}"
    else:
        final_answer = web_answer
        similarity_score_str = f"\n\n\nBased on Website data with similarity score: {web_score:.2f}\nPDF score: {pdf_score:.2f}"

    msg.content = f"{final_answer}"
    await msg.update()






# # Welcome to SIT Chatbot! 🚀🤖

# Hi there, Developer! 👋 We're excited to have you on board. Chainlit is a powerful tool designed to help you prototype, debug and share applications built on top of LLMs.

# ## Useful Links 🔗

# - **Documentation:** Get started with our comprehensive [Chainlit Documentation](https://docs.chainlit.io) 📚
# - **Discord Community:** Join our friendly [Chainlit Discord](https://discord.gg/k73SQ3FyUh) to ask questions, share your projects, and connect with other developers! 💬

# We can't wait to see what you create with Chainlit! Happy coding! 💻😊

# ## Welcome screen

# To modify the welcome screen, edit the `chainlit.md` file at the root of your project. If you do not want a welcome screen, just leave this file empty.

