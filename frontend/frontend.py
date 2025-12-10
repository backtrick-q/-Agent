import streamlit as st
import requests
import uuid
import os
from streamlit import spinner

#后端服务的地址
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
st.set_page_config(
    page_title = "AI命理师",
    page_icon = "🔮",
    layout = "centered"
)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
def reset_chat():
    try:
        if "session_id" in st.session_state.session_id:
            requests.post(
                f"{BACKEND_URL.replace('/chat','/delete_history')}",
                json = {"query":"delete","session_id":st.session_state.session_id}
            )
    except Exception as e:
        print(f"删除失败,但不影响重置,1小时之后您的历史对话记录将被自动删除")
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
with st.sidebar:
    st.header("⚙️ 控制台")
    st.text(f"ID: {st.session_state.session_id}")
    st.button(
        "🧹 清除对话历史",
        on_click = reset_chat,
        help = "点击开启新的对话,旧的对话将遗忘"
    )
    st.divider()
    st.subheader("📚 传授知识 (RAG)")
    with st.expander("有缘人,请给我更多文章助我看破天机"):
        url_input = st.text_input("输入文章URL",placeholder="例如百度百科链接...")
        if st.button("开始学习"):
            if url_input:
                with st.spinner("大师正在研读古籍..."):
                    try:
                        full_url = f"{BACKEND_URL}/add_urls"
                        res = requests.post(full_url,json={"url":url_input})
                        if res.status_code == 200:
                            st.success(f"学习成功 {res.json().get('detail')}")
                        else:
                            st.error(f"学习失败:{res.text}")
                    except Exception as e:
                        st.error(f"网络错误:{e}")
            else:
                st.warning("请先输入链接")
st.title("🔮命里有时终须有，命里无时莫强求")
st.caption("给我一枚铜钱,留下你的生辰八字,你会得到答案")
if "messages" not in st.session_state:
    st.session_state.messages = []
#渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("有缘人,请告诉老夫你的生辰八字..."):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.status("🔮 老夫掐指一算...",expanded=True) as status:
            try:
                st.write("...")
                payload = {
                    "query":prompt,
                    "session_id":st.session_state.session_id
                }
                full_url = f"{BACKEND_URL}/chat"
                response = requests.post(full_url,json=payload)
                print(f"请求地址:{full_url},状态码:{response.status_code}")
                #显示展示后端返回的状态码
                st.write(f"连接状态:{response.status_code}")
                if response.status_code == 200:
                    status.update(label="测算成功",state="complete",expanded=False)
                else:
                    status.update(label="测算失败",state="error")
                    st.error(f"小友乃大命数之人,吾竟然看不破:{response.text}")
            except Exception as e:
                status.update(label="网络异常",state="error")
                st.error(f"网络连接错误:{str(e)}")
        if response and response.status_code == 200:
            result = response.json()
            answer = result["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role":"assistant","content":answer})
            with st.expander("🕵️ 查看原始数据(Debug)"):
                st.json(result)