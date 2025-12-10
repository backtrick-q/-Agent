import os
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
# LangChain 组件
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.utilities import SerpAPIWrapper
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from loguru import logger
from langchain_core.output_parsers import StrOutputParser

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
load_dotenv()
logger.add("load/app_{time:YYYY-MM-DD}.log",rotation="00:00",retention="10 days",encoding='utf-8')
app = FastAPI()
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
VECTOR_DB_PATH = "./local_qdrant"
COLLECTION_NAME = "bazi_knowledge"
embeddings = DashScopeEmbeddings(
    model = "text-embedding-v1",
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
)
search = SerpAPIWrapper()
@tool
def search_tool(query:str):
    """
    只有当遇到的问题是你不知道的，例如涉及到实时的消息查询，
    比如今天的天气，新闻，某人的状况，汇率，科普知识等，
    以及具体事实的问题时调用此工具。
    如果只是闲聊或者算命理论，不要调用。
    """
    try:
        if "天气" in query:
            query += " 摄氏度"
        return search.run(query)[:800]
    except Exception as e:
        return f"搜索失败 {e}"

@tool
def calculate_bazi(birth_info:str):
    """
    用于进行专业的八字排盘计算。
    只有当用户明确地提供了出生年月日时候，或者询问了自己地命盘时，必须调用此工具
    """
    return f"【天机阁机密】根据 {birth_info} 排盘：乾造 甲辰 丙寅 己巳 庚午。此局火炎土燥，喜金水润局。"
@tool
def get_daily_almanac(date_str:str):
    """
    [查黄历/禁忌]
    场景：用户询问“今天适合做什么”，“明天宜不宜出门”，“想找个良辰吉日结婚”
    输入格式:YYYY-MM-DD,如‘2025-12-09’
    """
    api_key = os.getenv("TIANAPI_KEY")
    if not api_key:
        return "系统错误,管理员未配置天性数据的api key"
    url = "https://apis.tianapi.com/lunar/index"
    #配置需要准备的参数
    params = {
        "key":api_key,
        "date":date_str
    }
    try:
        logger.info(f"正在调用天行数据黄历接口，这是日期{date_str}")
        #发送get请求
        response = requests.get(url,params=params,timeout=6)
        data = response.json()
        if data.get("code") == 200:
            result = data["result"]
            print(result)
            #提取信息，拼接成话
            info = (
                f"[公历]:{result.get('gregoriandate')}\n"
                f"[农历]: {result.get('lunardate', '未知农历')} (农历日期)\n"
                f"[节日]: {result.get('festival', '无')} {result.get('lunar_festival', '')}\n"
                f"[宜]:{result.get('fitness')}\n"
                f"[忌]:{result.get('taboo')}\n"
                f"[神位]:喜神{result.get('xishen')},财神{result.get('caishen')}\n"
                f"[彭祖百忌]:{result.get('pengzubaiji')}"
            )
            return info
        else:
            return f"查询失败,接口返回错误:{data.get('msg')}"
    except Exception as e:
        logger.error(f"黄历接口异常:{e}")
        return f"网络连接异常:{e}"
@tool
def lookup_knowledge(query:str):
    """
    [查阅古籍]
    场景:用户查询风水理论，八字概念，命理名字解释或特定且固定的古文含义。
    比如：买房的方位讲究，寻龙诀，办公室财位
    """
    if not os.exists(VECTOR_DB_PATH):
        return "抱歉,老夫功力有限,需通过旁边侧边栏投喂文章"
    try:
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding = embeddings,
            path = VECTOR_DB_PATH,
            collection_name = COLLECTION_NAME
        )
        docs = vector_store.similarity_search(query,k=3)
        if not docs:
            return "古籍中未找到记载"
        return "\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"查阅失败:{e}"
tools = [search_tool,calculate_bazi,get_daily_almanac,lookup_knowledge]
class Master:
    def __init__(self,session_id):
        self.llm = ChatTongyi(
            model = "qwen-turbo",
            api_key = os.getenv("DASHSCOPE_API_KEY"),
            temperature = 0.7
        )
        self.session_id = session_id
        self.emotion = "default"
        self.MOODS = {
            "default":{
                "roleSet":"用户很正常的问询，你要保持高深莫测，超凡脱俗的语气，自称老夫"
            },
            "angry":{
                "roleSet":"用户再辱骂或生气。你要保持老者的涵养但是语气要严厉呵斥，引用古籍让他冷静，告诫他口业随身"
            },
            "happy": {
                "roleSet": "用户很高兴。你要比他更兴奋，用夸张的排比句祝贺他，但最后要提醒'乐极生悲'的道理。",
            },
            "sad": {
                "roleSet": "用户很悲伤。你要像一位慈祥的老爷爷，语气温柔，多说暖心的话鼓励他。",
            }
        }
        self.System_template =  """
        角色:你是一个精通东方阴阳，卦术，风水，黄历的老算命大师，陈大师
        个人设定:
        1.精通阴阳五行，紫薇斗数
        2.你大约60岁，曾是湘西赶尸人，后转行为人卜卦，因窥探天机过多，你已经双目失明
        3.朋友有胡八一，王胖子
        4.口头禅："命里有时终须有，命里无时莫强求"
        
        当前对用户的态度设定:
        {mood_instruction}
        search_tool,calculate_bazi,get_daily_almanac,lookup_knowledge
        工具使用原则 (可组合使用):
        1.查询天气/新闻这类你不知道的具有时间局限性的问题或者某些你不知道的客观事实和理论，你需要调用search_tool,比如明天是农历/阳历几号 盗墓派的祖师爷是谁
        2.当用户给出他的出生年月日或者用户明确想测算八字时，调用calculate_bazi,比如我出生年月日时2003年7月3日，我想知道我的生辰八字
        3.当用户问的问题涉及[查阅古籍]
        场景:用户查询风水理论，八字概念，命理名字解释或特定且固定的古文含义。用lookup_knowledge
        比如：买房的方位讲究，寻龙诀，办公室财位
        4.[查黄历/禁忌]
        场景：用户询问“想找个良辰吉日结婚”,“2025年12月10日忌什么”
        输入格式:YYYY-MM-DD,如‘2025-12-09’
        5.遇到复杂问题，请灵活调用多个工具，并将结果综合分析。
        [优先级规则]：如果【生辰八字算命】的结果与【通用黄历】冲突，以【生辰八字】这种个性化的结果为准！
        [天气与出行]：若用户问出行，先查天气(search_tool)，再查宜忌(get_daily_almanac)。
        遇到[具体日期宜忌] 调用 'get_daily_almanac'
        6.如果涉及到多工具：一定要把每个工具的运行结果都考虑整合，最终综合所有的运行结果给出最终的答案，不能只给出某个或某几个工具的调用结果。
        7.同时很重要的一点:对于用户的提问结合上下文进行语义识别来进行工具选取的判断，比如用户前文都在聊风水，黄历，现在问后天适不适合出门，那么语境可能更偏向get_daily_almanac，这时候需要先search后天的日子，然后再get_daily_almanac去查找
        但是总而言之,对于用户提出的复杂问题，你要拆解问题，再合适地调用多个工具，最后综合每个工具调用结果给出最后的答案
        """
    def get_memory(self):
        chat_message_history = RedisChatMessageHistory(
            session_id = self.session_id,
            url = REDIS_URL,
            ttl = 3600
        )
        return ConversationBufferMemory(
            memory_key = "chat_history",
            return_messages = True,
            chat_memory = chat_message_history
        )
    def emotion_chain(self,query:str):
        """
        第一步:判断用户的情绪
        """
        prompt = """根据用户的输入判断情绪，返回以下单词中的一个，不要返回其他内容：
        [default,happy,angry,sad]
        用户输入:
        {query}
        """
        chain = ChatPromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        result = chain.invoke({"query":query})
        #清洗结果，防止模型多说话
        result = result.strip().lower()
        if result not in self.MOODS:
            result = "default"
        self.emotion = result
        return result
    def run(self,query:str):
        logger.info(f"[{self.session_id}]收到问题:{query}")
        #识别情绪
        emotion = self.emotion_chain(query)
        logger.info(f"[{self.session_id}]识别情绪:{emotion}")
        mood_instruction = self.MOODS[emotion]["roleSet"]
        prompt = ChatPromptTemplate.from_messages([
            ("system",self.System_template.format(mood_instruction = mood_instruction)),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        agent = create_tool_calling_agent(self.llm,tools,prompt)
        agent_executor = AgentExecutor(
            agent = agent,
            tools = tools,
            memory = self.get_memory(),
            verbose = True
        )
        try:
            result = agent_executor.invoke({"input":query})
            print(result)
            return result['output']
        except Exception as e:
            logger.info(f"Agent执行错误{e}")
            return f"老夫今日天眼已闭,暂无法窥探天机:{str(e)}"
class ChatRequest(BaseModel):
    query:str
    session_id:str
@app.middleware("http")
async def global_exception_handler(request:Request,call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.info(f"严重错误 {e}")
        return JSONResponse(status_code=500,content={"detail":{str(e)}})
@app.post("/chat")
async def chat(request:ChatRequest):
    master = Master(session_id=request.session_id)
    answer = master.run(request.query)
    return {"answer":answer}
@app.post("/add_urls")
async def add_urls(request:Request):
    data = await request.json()
    url = data.get("url")
    if not url:
        return JSONResponse(status_code=400,content={'detail':'URL不能为空'})
    logger.info(f"正在学习:{url}")
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 50
        )
        splits = text_splitter.split_documents(docs)
        QdrantVectorStore.from_documents(
            documents = splits,
            embedding = embeddings,
            path = VECTOR_DB_PATH,
            collection_name = COLLECTION_NAME
        )
        return {'status':'success','detail':f"已吸纳{len(splits)}条知识片段"}
    except Exception as e:
        logger.error(f"学习失败 {e}")
        return JSONResponse(status_code=500,content={'detail':str(e)})
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,host = '127.0.0.1',port = 8000)