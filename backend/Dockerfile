# 1. 选底座：使用官方 Python 3.10 精简版
FROM python:3.10-slim

# 2. 设工作区：在容器里创建一个叫 /app 的文件夹，并进入
WORKDIR /app

# 3. 设环境变量：
# 防止 Python 生成 .pyc 缓存文件
ENV PYTHONDONTWRITEBYTECODE=1
# 防止 Python 缓冲输出 (保证你能立刻看到 print 日志)
ENV PYTHONUNBUFFERED=1

# 4. 装依赖：
# 先只复制 requirements.txt (利用 Docker 缓存机制加速构建)
COPY requirements.txt .
# 安装库，使用国内清华源加速，防止超时
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 搬代码：把当前文件夹下的所有文件，复制到容器的 /app 里
COPY . .

# 6. 开端口：告诉 Docker 我这个服务用 8000 端口
EXPOSE 8000

# 7. 启动命令：
# 注意 host 必须是 0.0.0.0，不能是 127.0.0.1，否则外部访问不到
CMD ["uvicorn", "back:app", "--host", "0.0.0.0", "--port", "8000"]