# 1. 选底座
FROM python:3.10-slim

# 2. 设工作区
WORKDIR /app

# 3. 装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 搬代码
COPY . .

# 5. 开端口 (Streamlit 默认端口)
EXPOSE 8501

# 6. 启动命令
CMD ["streamlit", "run", "frontend.py", "--server.address=0.0.0.0", "--server.port=8501"]