import chainlit as cl
from langchain_community.embeddings import OpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
from passlib.hash import bcrypt
from langchain.memory import ChatMessageHistory, ConversationBufferMemory


USER_DB = {
    "Shreya": bcrypt.hash("ZaQ1@wSx"),  # Hash the password for security
    "user": bcrypt.hash("qwerty123")
}

embeddings = OpenAIEmbeddings()
embedding_model_for_pdf = OpenAIEmbeddings(model="text-embedding-3-large")
llm=ChatOpenAI(temperature=0, model_name='gpt-4o', streaming=True)

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Validates username and password"""
    
    # Check if user exists
    if username in USER_DB:
        # Verify password hash
        if bcrypt.verify(password, USER_DB[username]):  
            return cl.User(
                identifier=username,
                metadata={"role": "admin" if username == "Shreya" else "user", "provider": "credentials"}
            )
    
    return None  # Authentication failed


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Ask questions about data in PDFs",
            markdown_description="This chatbot can answer questions based on the PDFs on SIT website.",
            icon="https://img.icons8.com/?size=100&id=m31DrURYH9au&format=png&color=f5145f",
        ),
        cl.ChatProfile(
            name="Ask questions about data on website",
            markdown_description="This chatbot can answer questions based on the website on SIT website.",
            icon="https://img.icons8.com/?size=100&id=m31DrURYH9au&format=png&color=f5145f",
        ),
    ]

@cl.on_chat_start
async def on_chat_start():
    user = cl.user_session.get("user")
    # if user:
    #     await cl.Message(content=f"Welcome, {user.identifier}! You are logged in as {user.metadata['role']}").send()
    # else:
    #     await cl.Message(content="Authentication failed!").send()
    chat_profile = cl.user_session.get("chat_profile")
    await cl.Message(
        content=f"Hi {user.identifier}, how may I help you today? \nYou can {chat_profile}."
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    current_profile = cl.user_session.get("chat_profile")
    cb = cl.AsyncLangchainCallbackHandler()
    # print("current_profile: ", current_profile)
    if current_profile and current_profile == "Ask questions about data in PDFs":
        vector_db_path = r"C:\Shreya\Final Year Project\chainlit-chatbot\Faiss_vectorDB_PDF_data"
        VectorStore_pdf = FAISS.load_local(vector_db_path, embedding_model_for_pdf, allow_dangerous_deserialization=True)
        chain_pdf = RetrievalQA.from_chain_type(llm=llm, retriever=VectorStore_pdf.as_retriever())
        # answer = chain_pdf.invoke({"query": message.content})["result"]
        res = await chain_pdf.acall(message.content, callbacks=[cb])
        answer = res["result"]

    elif current_profile and current_profile == "Ask questions about data on website":
        vector_db_path = r"C:\Shreya\Final Year Project\chainlit-chatbot\Faiss_vectorDB_website_data"
        VectorStore = FAISS.load_local(vector_db_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        
        chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=VectorStore.as_retriever())
        # response = chain.invoke(message.content)
        response = await chain.acall(message.content, callbacks=[cb])
        # print(f"res keys: ", response.keys())
        answer = response['answer']
        

    else:
        vector_db_path = None
    
    # print("PATH: ", vector_db_path)
    
    if vector_db_path:
        # await cl.Message(content=f"Using vector DB at: {vector_db_path}").send()
        await cl.Message(content=answer).send()
    else:
        await cl.Message(content="No valid profile selected.").send()





# # Welcome to SIT Chatbot! 🚀🤖

# Hi there, Developer! 👋 We're excited to have you on board. Chainlit is a powerful tool designed to help you prototype, debug and share applications built on top of LLMs.

# ## Useful Links 🔗

# - **Documentation:** Get started with our comprehensive [Chainlit Documentation](https://docs.chainlit.io) 📚
# - **Discord Community:** Join our friendly [Chainlit Discord](https://discord.gg/k73SQ3FyUh) to ask questions, share your projects, and connect with other developers! 💬

# We can't wait to see what you create with Chainlit! Happy coding! 💻😊

# ## Welcome screen

# To modify the welcome screen, edit the `chainlit.md` file at the root of your project. If you do not want a welcome screen, just leave this file empty.

