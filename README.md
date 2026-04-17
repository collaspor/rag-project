# HW-Chat: 高性能 RAG 商业知识库系统

HW-Chat 是一款基于 **LangChain** 和 **FastChat** 构建的高性能检索增强生成（RAG）商业知识库系统。本项目集成了文档处理、向量检索、大模型对话及自动化评估等核心功能，旨在为企业提供稳健、可扩展且私有化的知识库解决方案。

---

## 🌟 核心特性

- **多模态文档处理**：支持 PDF、TXT 等多种格式文档的自动化解析与高性能切片。
- **高性能向量检索**：基于 **FAISS** 的本地向量库管理，支持多线程并发访问与 LRU 缓存机制。
- **灵活的模型调度**：集成 **FastChat** 框架，支持本地 GPU 模型（如 ChatGLM3-6b）与在线 API（如 智谱 AI）的统一调度。
- **深度中文优化**：内置针对中文语境优化的递归字符切分器与标题增强技术。
- **完整的 RAG 评估体系**：独立的 `hwrag` 模块，支持对检索准确率、回答质量等指标进行自动化跑分。
- **会话持久化**：基于 SQLAlchemy 实现用户信息、对话历史与知识库元数据的持久化存储。
- **前后端一体化**：后端基于 **FastAPI**，并自动托管 Vue.js 构建的前端界面。

---

## 🏗️ 技术架构

- **核心框架**：[LangChain v0.2.x](https://github.com/langchain-ai/langchain)
- **模型后端**：[FastChat](https://github.com/lm-sys/FastChat)
- **Web 框架**：FastAPI
- **向量数据库**：FAISS
- **关系型数据库**：SQLAlchemy (支持 SQLite/MySQL)
- **前端技术**：Vue.js (静态托管)

---

## 📁 项目结构

```text
hw-chat/
├── configs/            # 全局配置中心（模型、知识库、服务器参数）
├── server/             # 后端核心业务逻辑（对话、数据库、知识库管理）
├── hwrag/              # RAG 自动化评估与流水线模块
├── document_loaders/   # 增强型文档加载器
├── text_splitter/      # 针对中文优化的文本切分策略
├── playground/         # 技术实验室与原型验证脚本
├── scripts/            # 数据预处理与运维工具
├── static/             # 前端编译产物静态文件夹
├── startup.py          # 项目一键启动入口
└── requirements.txt    # 项目依赖列表
```

---

## 🚀 快速启动

### 1. 环境准备
确保已安装 Python 3.8+ 环境，并安装相关依赖：
```bash
pip install -r requirements.txt
```

### 2. 参数配置
在 `configs/` 目录下根据实际环境修改配置：
- `configs/model_config.py`：配置 LLM 和 Embedding 模型路径。
- `configs/server_config.py`：配置服务器端口（默认 6006）。

### 3. 启动服务
运行启动脚本，一键开启 Controller、Worker 和 API Server：
```bash
python startup.py -m
```

---

## 接口说明 (API Usage)

### 知识库问答接口
**Endpoint**: `POST /api/chat/knowledge_base_chat`

**请求示例**:
```json
{
    "query": "什么是 GLM4 多角色对话？",
    "user_id": "admin",
    "conversation_id": "unique-uuid-string",
    "knowledge_base_name": "private",
    "top_k": 3,
    "score_threshold": 0.5,
    "stream": false,
    "model_name": "chatglm3-6b"
}
```

---

## 📊 评估模块 (HWRAG)

本项目内置了 `hwrag` 评估模块，开发者可以对 RAG 的各个环节进行性能打分。
- 路径：`hwrag/evaluator/`
- 支持指标：检索召回率、答案相关性等。

---

## 📄 开源协议
本项目遵循 MIT 开源协议。
