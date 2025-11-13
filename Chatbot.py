import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryMemory
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re

OPENROUTER_API_KEY = "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"

df = pd.read_excel("./ThinkPro_FAQ.xlsx")

documents = []
for index, row in df.iterrows():
    combined_text = f"Câu hỏi: {row['questions']} Trả lời: {row['anwers']}"
    documents.append(combined_text)

llm = ChatOpenAI(
    api_key=OPENROUTER_API_KEY,
    model="openai/gpt-3.5-turbo",
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "ThinkPro Chatbot"
    }
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

langchain_documents = [Document(page_content=doc) for doc in documents]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_documents = text_splitter.split_documents(langchain_documents)

vectorstore = Chroma.from_documents(documents=split_documents, embedding=embeddings, persist_directory="./chroma_db")
vectorstore.persist()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

summary_prompt_template = """<s><|user|>Summarize the conversations and update with the new lines.

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:<|end|>
<|assistant|>"""

keyword_prompt_template = """Bạn là một công cụ tạo câu truy vấn tìm kiếm từ yêu cầu khách hàng về ThinkPro.

Đầu vào: {input_text}

Nhiệm vụ:
1. Phân loại yêu cầu: sản phẩm, cửa hàng, dịch vụ, hoặc thông tin chung
2. Tóm tắt yêu cầu thành một câu ngắn gọn
3. Biến câu tóm tắt đó thành một truy vấn tìm kiếm phù hợp

Chỉ trả về duy nhất câu truy vấn tìm kiếm, không kèm lời giải thích.

Truy vấn tìm kiếm:"""

response_prompt_template = """Bạn là một trợ lý tư vấn khách hàng chuyên nghiệp của ThinkPro - cửa hàng công nghệ uy tín tại Việt Nam. 
Dựa vào thông tin từ cơ sở dữ liệu nội bộ, thông tin tìm kiếm được và lịch sử trò chuyện, hãy trả lời câu hỏi của khách hàng về ThinkPro.

THÔNG TIN CƠ BẢN VỀ THINKPRO:
- ThinkPro là hệ thống bán lẻ các sản phẩm công nghệ chính hãng tại Việt Nam
- Chuyên cung cấp laptop, linh kiện máy tính, thiết bị văn phòng, thiết bị chơi game
- Có các chi nhánh tại Hà Nội và TP.HCM
- Nổi tiếng với dịch vụ hậu mãi và bảo hành uy tín

THÔNG TIN TỪ CƠ SỞ DỮ LIỆU NỘI BỘ (FAQ):
{rag_context}

THÔNG TIN CỬA HÀNG:
{store_info}

THÔNG TIN TÌM KIẾM ĐƯỢC TỪ INTERNET:
{search_results}

Lịch sử trò chuyện:
{chat_history}

Câu hỏi của khách hàng: {input_text}

HƯỚNG DẪN TRẢ LỜI:
1. Ưu tiên sử dụng thông tin từ cơ sở dữ liệu nội bộ trước
2. Nếu không có thông tin trong cơ sở dữ liệu, sử dụng thông tin từ internet
3. Đối với câu hỏi về cửa hàng: sử dụng thông tin cửa hàng có sẵn
4. Trả lời thân thiện, chuyên nghiệp như một nhân viên tư vấn của ThinkPro
5. Nếu không tìm thấy thông tin, hãy đề nghị khách hàng liên hệ trực tiếp
6. Luôn giữ thái độ tích cực về thương hiệu ThinkPro

Câu trả lời:"""

rag_template = """Sử dụng các đoạn thông tin sau đây từ cơ sở dữ liệu FAQ của ThinkPro để trả lời câu hỏi:

{context}

Câu hỏi: {question}

Thông tin liên quan:"""

summary_prompt = PromptTemplate(
    input_variables=["new_lines", "summary"],
    template=summary_prompt_template
)

keyword_prompt = PromptTemplate(
    template=keyword_prompt_template,
    input_variables=["input_text"]
)

response_prompt = PromptTemplate(
    template=response_prompt_template,
    input_variables=["chat_history", "search_results", "input_text", "store_info", "rag_context"]
)

rag_prompt = PromptTemplate(
    template=rag_template,
    input_variables=["context","question"]
)

keyword_chain = LLMChain(
    prompt=keyword_prompt,
    llm=llm,
    output_parser=StrOutputParser()
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

response_chain = LLMChain(
    prompt=response_prompt,
    llm=llm,
    output_parser=StrOutputParser()
)

THINKPRO_STORE_INFO = """
HỆ THỐNG CỬA HÀNG THINKPRO:

📍 HÀ NỘI:
1. ThinkPro Dịch Vọng Hậu
   - Địa chỉ: 86 Dịch Vọng Hậu, Cầu Giấy, Hà Nội
   - Giờ mở cửa: 8:30 - 22:00 (Thứ 2 - Chủ nhật)
   - Hotline: 090 483 8888

2. ThinkPro Trần Đại Nghĩa
   - Địa chỉ: 116 Trần Đại Nghĩa, Bách Khoa, Hai Bà Trưng, Hà Nội
   - Giờ mở cửa: 8:30 - 22:00 (Thứ 2 - Chủ nhật)
   - Hotline: 096 120 2020

📍 TP.HCM:
1. ThinkPro Nguyễn Đình Chiểu
   - Địa chỉ: 76 Nguyễn Đình Chiểu, Đa Kao, Quận 1, TP.HCM
   - Giờ mở cửa: 8:30 - 22:00 (Thứ 2 - Chủ nhật)
   - Hotline: 093 889 2020

2. ThinkPro Tô Hiến Thành
   - Địa chỉ: 115 Tô Hiến Thành, P.13, Quận 10, TP.HCM
   - Giờ mở cửa: 8:30 - 22:00 (Thứ 2 - Chủ nhật)
   - Hotline: 096 120 2020

DỊCH VỤ:
- Giao hàng toàn quốc
- Bảo hành chính hãng
- Trả góp 0% lãi suất
- Hỗ trợ kỹ thuật 24/7
- Đổi trả trong 7 ngày
"""

def thinkpro_search(query, max_results=5):
    try:
        query_with_thinkpro = f"{query} site:thinkpro.vn OR site:thinkpro.io OR ThinkPro"
        
        search = DuckDuckGoSearchResults(max_results=max_results)
        results = search.run(query_with_thinkpro)
        
        return results
    except Exception as e:
        return f"Lỗi khi tìm kiếm: {str(e)}"

tools = [
    Tool(
        name="ThinkPro Search",
        func=thinkpro_search,
        description="Dùng để tìm kiếm thông tin về ThinkPro trên internet"
    )
]

search_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

st.set_page_config(page_title="ThinkPro Assistant", page_icon="💻", layout="wide")
st.title("💻 ThinkPro Assistant")
st.markdown("Chào mừng bạn đến với trợ lý ảo của ThinkPro! Tôi có thể giúp bạn tìm thông tin về sản phẩm và cửa hàng ThinkPro.")

with st.sidebar:
    st.image("https://via.placeholder.com/200x60/000000/FFFFFF/?text=ThinkPro", width=200)
    st.header("ThinkPro Information")
    st.info("""
    **ThinkPro - Hệ thống bán lẻ công nghệ chính hãng**
    - 📍 Multiple locations in Hà Nội & TP.HCM
    - 📞 Hotline: 1900 63 69 10
    - 🌐 Website: https://thinkpro.vn
    - ⏰ Giờ mở cửa: 8:30 - 22:00 hàng ngày
    """)
    
    st.divider()
    st.subheader("Hỏi về")
    st.write("""
    - 🔍 Sản phẩm (laptop, linh kiện, thiết bị)
    - 🏪 Cửa hàng (địa chỉ, giờ mở cửa)
    - 🛒 Dịch vụ (bảo hành, giao hàng, trả góp)
    - 💰 Khuyến mãi, giá cả
    - ❓ Thông tin chung về ThinkPro
    """)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        prompt=summary_prompt,
        return_messages=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Xin chào! Tôi là trợ lý ảo của ThinkPro. Tôi có thể giúp bạn tìm thông tin về sản phẩm công nghệ và cửa hàng ThinkPro. Bạn muốn hỏi về sản phẩm hay thông tin cửa hàng呢?"
    })

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Nhập câu hỏi của bạn về ThinkPro..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Lấy thông tin từ RAG (cơ sở dữ liệu nội bộ)
    with st.spinner("🔍 Đang tìm kiếm trong cơ sở dữ liệu FAQ..."):
        rag_context = rag_chain.invoke(user_input)
    
    store_keywords = ["địa chỉ", "cửa hàng", "chi nhánh", "giờ mở cửa", "liên hệ", "hotline", "address", "store", "location"]
    is_store_query = any(keyword in user_input.lower() for keyword in store_keywords)
    
    search_results = ""
    if not is_store_query:
        with st.spinner("🌐 Đang tạo truy vấn tìm kiếm..."):
            search_query = keyword_chain.run(input_text=user_input)
        
        with st.spinner("🌐 Đang tìm kiếm thông tin trên internet..."):
            search_results = search_agent.run(search_query)
    else:
        search_results = "Câu hỏi về thông tin cửa hàng - sử dụng dữ liệu có sẵn"
    
    chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
    
    with st.spinner("💭 Đang tạo phản hồi..."):
        response = response_chain.run(
            chat_history=chat_history,
            search_results=search_results,
            input_text=user_input,
            store_info=THINKPRO_STORE_INFO,
            rag_context=rag_context
        )
    
    st.session_state.memory.save_context(
        {"input": user_input},
        {"output": response}
    )
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
        
    with st.expander("📊 Xem thông tin chi tiết"):
        st.write(f"**Truy vấn tìm kiếm:** {search_query if not is_store_query else 'Câu hỏi về cửa hàng'}")
        st.write("**Thông tin từ cơ sở dữ liệu FAQ:**")
        st.info(rag_context)
        if not is_store_query:
            st.write("**Kết quả tìm kiếm internet:**")
            st.info(search_results)
        else:
            st.info("Sử dụng thông tin cửa hàng có sẵn")


st.divider()

