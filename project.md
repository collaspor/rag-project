# 项目概况：HW-Chat 高性能 RAG 商业知识库系统

> 本文档由对整个代码库的系统性梳理生成，用于建立项目全局认知，为后续维护与改造提供参考。

---

## 1. 项目定位

HW-Chat 是一套基于 **LangChain + FastChat** 的检索增强生成（RAG）知识库问答系统，提供：

- 多格式文档解析、切片、向量化入库
- 基于 FAISS 的向量检索 + 可选 Reranker 重排
- 本地 GPU 模型（ChatGLM3-6B）与在线 API（智谱 GLM-4）统一调度
- 用户 / 会话 / 消息 / 知识库元数据持久化
- 独立的 RAG 自动化评估模块（`hwrag`）
- FastAPI 后端 + Vue 静态前端一体化部署

---

## 2. 技术栈

| 层 | 技术 |
|----|------|
| Web 框架 | FastAPI 0.111 + Uvicorn + SSE（流式输出） |
| RAG 框架 | LangChain 0.2.x（community / core / openai / text-splitters） |
| 模型后端 | FastChat 0.2.35（Controller + ModelWorker + OpenAI API） |
| 向量库 | FAISS（faiss-cpu 1.8），抽象层预留 Milvus/Zilliz/PG/ES/ChromaDB |
| 关系库 | SQLAlchemy 2.0 异步 ORM + asyncmy（MySQL，charset utf8mb4） |
| Embedding | bge-large-zh-v1.5（本地）/ 智谱 embedding-2（在线） |
| Reranker | bge-reranker-large（sentence-transformers CrossEncoder） |
| 文档解析 | unstructured / pdfplumber / pypdf 等 |
| 前端 | Vue.js 编译产物（`static/dist`，由后端静态托管） |
| 评估 | 自研 `hwrag`（借鉴 FlashRAG），部分组件仍依赖 `flashrag` |

> Python 3.8+；目标部署环境为 AutoDL（Linux + CUDA），配置中存在 `/root/autodl-tmp/...` 等 Linux 绝对路径。

---

## 3. 目录结构总览

```text
rag-project/
├── configs/            # 全局配置中心
├── server/             # 后端核心业务逻辑
├── hwrag/              # RAG 自动化评估模块（离线评测）
├── document_loaders/   # 增强型文档加载器
├── text_splitter/      # 中文优化的文本切分策略
├── playground/         # 技术实验 / 原型脚本（约 95 个文件，非生产代码）
├── scripts/            # 数据预处理脚本（preprocess_wiki.py）
├── data/               # 原始 / 解析数据（PDF、json、txt）
├── knowledge_base/     # 知识库存储（private / test / wiki，含 FAISS index）
├── static/             # 旧版 Vue 前端编译产物（dist，无源码）
├── frontend/           # 新版自包含管理前端（免构建单页 HTML，后端挂载于 /app）
├── startup.py          # 一键启动入口（Controller + Worker + API Server）
├── requirements.txt    # 依赖清单
├── README.md
└── project.md          # 本文档
```

---

## 4. 配置中心（configs/）

| 文件 | 作用 | 关键项 |
|------|------|--------|
| `basic_config.py` | 日志格式、日志路径（`logs/`）、临时目录（系统 tmp/chatchat） | `LOG_FORMAT`、`LOG_PATH`、`BASE_TEMP_DIR` |
| `model_config.py` | 模型路径与参数 | `LLM_MODELS=["chatglm3-6b","zhipu-api"]`、`EMBEDDING_MODEL=bge-large-zh-v1.5`、`USE_RERANKER=True`、`RERANKER_TOP_K=3`、`VECTOR_SEARCH_TOP_K=5`、`ONLINE_LLM_MODEL`（zhipu/openai） |
| `server_config.py` | 各服务端口 | API `6006`、Controller `20001`、OpenAI API `20000`、ModelWorker `20002`、zhipu-api `21001`，默认 host `127.0.0.1` |
| `kb_config.py` | 知识库与切分参数 | `SQLALCHEMY_DATABASE_URI`（MySQL）、`DEFAULT_VS_TYPE=faiss`、`CHUNK_SIZE=250`、`OVERLAP_SIZE=50`、`TEXT_SPLITTER_NAME=ChineseRecursiveTextSplitter`、`KB_ROOT_PATH`、`kbs_config`、`text_splitter_dict` |
| `prompt_config.py` | 提示词模板 | `llm_chat`（default/with_history）、`knowledge_base_chat`（default/text/empty） |

> **凭证管理**：API Key、数据库密码均通过 `os.getenv(...)` 读取（如 `ZHIPUAI_API_KEY`、`HW_CHAT_DB_PASSWORD`），默认值为占位符。生产环境必须以环境变量注入。

---

## 5. 后端核心（server/）

### 5.1 应用入口与工具

| 文件 | 职责 |
|------|------|
| `api_router.py` | **真正的应用入口**：`create_app()` 创建 FastAPI、配置 CORS、`mount_app_routes()` 挂载路由、托管 `static/dist` 前端 |
| `utils.py` | 核心工具库：模型地址解析、`get_ChatOpenAI`、`get_model_worker_config`、`get_prompt_template`、`load_local_embeddings`、设备检测、统一响应模型 `BaseResponse`/`ListResponse` |
| `embeddings_api.py` | 文本向量化接口：`embed_texts`/`aembed_texts`、`embed_documents`、FastAPI 端点 |
| `minx_chat_openai.py` | `MinxChatOpenAI`，用 tiktoken 兼容非 OpenAI 模型的 token 编码 |
| `main.py` | ⚠️ **名不副实**：实为 BM25/检索器实验脚本（pyserini），未被应用引用 |

### 5.2 API 路由（已挂载）

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/chat` | 通用大模型对话（SSE 流式） |
| POST | `/api/chat/knowledge_base_chat` | 知识库 RAG 对话（SSE 流式，返回 answer + 来源 docs） |
| POST | `/api/conversations` | 新建会话 |
| GET | `/api/users/{user_id}/conversations` | 获取用户会话列表 |
| GET | `/api/conversations/{conversation_id}/messages` | 获取会话消息列表 |
| GET | `/api/knowledge_base/list_files` | 获取知识库内文件列表（需 `user_id` 鉴权） |
| POST | `/api/knowledge_base/upload_docs` | 上传文件到知识库并可选向量化（需 `user_id` 鉴权） |
| POST | `/api/knowledge_base/delete_docs` | 删除知识库内指定文件（需 `user_id` 鉴权） |

> 已挂载的知识库管理接口（`list_files`/`upload_docs`/`delete_docs`）均要求传入 `user_id` 并经 `check_user` 校验。其调用链 `kb_doc_api → KBService → FaissKBService` 已完成异步化改造。
> `kb_doc_api.py` 中其余文档管理函数（`update_docs_by_id`/`update_info`/`download_doc`/`recreate_vector_store` 等）仍**未注册为路由**，且其依赖的部分 `KBService` 方法（`create_kb`/`clear_vs`/`drop_kb`/`update_info`/`list_docs`/`update_doc_by_ids` 等）仍存在同步调用 async repository 未 await 的问题，挂载前需先异步化。

### 5.3 对话逻辑（server/chat/）

- `chat.py`：通用对话。用户校验 → 写消息 → 注册流式 & 持久化回调 → 可选加载 DB 历史（`ConversationBufferDBMemory`）→ `LLMChain` → SSE 返回 `{text, message_id}`
- `knowledge_base_chat.py`：RAG 对话。取知识库服务 → 向量检索 → 可选 Reranker → 拼接 context → 套用模板 → `LLMChain` → SSE 返回 `{answer, docs}`
- `utils.py`：`History` 对话历史模型（支持 jinja2 模板转换）

### 5.4 数据库（server/db/）

**基础设施**：`base.py`（异步引擎/Session/Base）、`session.py`（`async_session_scope`、`with_async_session`、`get_async_db`）、`create_all_model(s).py`（建表，两文件重复）、`delete_all_table.py`（删表）

**数据模型（表）**：

| 表 | 模型 | 说明 |
|----|------|------|
| `user` | `UserModel` | 用户（username 唯一、password_hash） |
| `conversation` | `ConversationModel` | 会话（UUID、user_id FK、chat_type） |
| `message` | `MessageModel` | 消息（query/response、meta_data JSON、feedback） |
| `knowledge_base` | `KnowledgeBaseModel` | 知识库（kb_name、vs_type、embed_model、file_count） |
| `knowledge_file` | `KnowledgeFileModel` | 知识文件（loader/splitter/version/mtime/docs_count） |
| `file_doc` | `FileDocModel` | 文件分块文档（doc_id 对应向量库 ID） |

**仓储（repository/）**：知识库、知识文件、消息三类 CRUD 方法。

### 5.5 知识库管理（server/knowledge_base/）

- `kb_doc_api.py`：文档管理 API 函数集（search/upload/delete/update/download/recreate_vector_store，多线程保存）
- `utils.py`：路径工具、`validate_kb_name`（防路径穿越）、`LOADER_DICT`（.md/.json/.jsonl/.pdf）、`make_text_splitter`、`KnowledgeFile` 类、`files2docs_in_thread`
- `init_vs.py`：向量库初始化脚本（PDF pipeline / jsonl 入 FAISS，独立运行）
- `kb_service/base.py`：**核心抽象** `KBService` + `KBServiceFactory`（按 vs_type 工厂）+ `EmbeddingsFunAdapter`（向量归一化）+ `get_kb_details`
- `kb_service/faiss_kb_service.py`：`FaissKBService` 具体实现（do_search/do_add_doc/do_delete_doc/load_vector_store）
- `kb_cache/base.py`：`CachePool`（LRU）、`EmbeddingsPool`（嵌入模型缓存，单例 `embeddings_pool`）
- `kb_cache/faiss_cache.py`：`ThreadSafeFaiss`、`KBFaissPool`/`MemoFaissPool`（线程安全 + 磁盘加载，单例 `kb_faiss_pool`/`memo_faiss_pool`）
- `model/kb_document_model.py`：`DocumentWithVSId`（Document + id + score）

### 5.6 模型 Worker（server/model_workers/）

- `base.py`：FastChat `ApiModelWorker` 抽象基类 + 各类参数模型（预留 MiniMax/讯飞/Azure 字段）
- `zhipu.py`：`ChatGLMWorker` —— 智谱 GLM-4 在线 worker（JWT 鉴权、流式 chat、embedding-2）
- **当前仅实现智谱 AI**；OpenAI 在配置中预留但无独立 worker

### 5.7 辅助模块

| 模块 | 职责 |
|------|------|
| `reranker/reranker.py` | `LangchainReranker`：CrossEncoder 文档重排压缩器，取 top_n |
| `memory/conversation_db_buffer_memory.py` | `ConversationBufferDBMemory`：从 DB 读历史构建 Memory，含 token 剪枝 |
| `callback_handler/conversation_callback_handler.py` | LLM 结束时将回答持久化到 DB |
| `verify/check_user.py` | `check_user`：校验 user_id 是否存在（否则 401） |
| `verify/utils.py` | 会话/消息 API 处理函数 + Pydantic 响应模型 |

---

## 6. 文档加载与切分

### document_loaders/
- `interface.py`：`Pipeline` 抽象接口
- `pdfloader.py`：`UnstructuredLightPipeline`，基于 unstructured `partition_pdf`（hi_res + yolox），支持表格抽取，输出切分后的 Document

### text_splitter/
- `chinese_recursive_text_splitter.py`：`ChineseRecursiveTextSplitter`，按中文标点（。！？；，）递归切分，合并去除多余换行
- `zh_title_enhance.py`：中文标题增强（识别标题并与上级标题拼合，可由 `ZH_TITLE_ENHANCE` 开关控制）

---

## 7. 评估模块（hwrag/）

借鉴 FlashRAG 的离线评测闭环：`query → retriever → (reranker) → generator → evaluator`。

| 子目录 | 职责 |
|--------|------|
| `config/` | `config.py`（多层配置合并）+ `pipeline_eval.yaml`（实际运行配置） |
| `dataset/` | `Item` / `Dataset`（JSONL 加载、采样、保存） |
| `retriever/` | `BM25Retriever`（pyserini）、`DenseRetriever`（FAISS）、`Encoder`/`STEncoder`、`CrossReranker`/`BiReranker` |
| `generator/` | `EncoderDecoder`/`VLLM`/`HFCausalLM`/`FastChat` 四种后端 |
| `evaluator/` | `Evaluator`（反射注册指标）+ `metrics.py`（全部指标实现） |
| `pipeline/` | `BasicPipeline`/`SequentialPipeline`/`ConditionalPipeline`（含 `__main__` 入口） |
| `prompt/` | `PromptTemplate`（按模型类型套 chat template） |
| `utils/` | 组件工厂 + JSON→JSONL 转换脚本 |
| `data/test/` | 5 个中文 QA 测试集 jsonl |

**支持指标**：
- 生成质量：`em`、`sub_em`、`f1`、`precision`、`recall`、`rouge-1/2/l`、`bleu`、`input_tokens`
- 检索质量：`retrieval_recall`、`retrieval_precision`、`hit@1/3/5`、`mrr`
- 中文适配：jieba 分词 + token 重叠软匹配（阈值 0.6）

---

## 8. 启动流程（startup.py）

`python startup.py -m` 通过 `multiprocessing`（spawn）依次拉起：

1. **Controller**（FastChat 调度，端口 20001）
2. **OpenAI API Server**（兼容层，端口 20000）
3. **Model Worker**（本地 chatglm3-6b / 在线 zhipu-api）
4. **API Server**（业务后端，端口 6006）

支持运行期 `/release_worker` 动态切换/释放模型（start/stop/replace）。

---

## 9. 典型 RAG 请求链路

```
POST /api/chat/knowledge_base_chat
  → check_user 校验
  → add_message_to_db 写入用户消息
  → KBServiceFactory.get_service_by_name（查 DB 取 vs_type/embed_model）
  → FaissKBService.do_search（query 向量化 → FAISS 相似度检索 top_k）
  → [可选] LangchainReranker 重排到 top_n
  → 拼接 context + knowledge_base_chat 模板
  → LLMChain（ChatOpenAI 指向 FastChat / 智谱）
  → SSE 流式返回 {answer, docs（含来源下载链接）}
  → ConversationCallbackHandler 持久化模型回答
```

---

## 10. 已识别的风险与待改进点

> 后续改造时优先关注：

1. **CORS 全开放**：`allow_origins=["*"]` + `allow_credentials=True`，生产环境存在 CSRF / 凭证泄露风险。
2. **认证授权薄弱**：`check_user` 仅校验 user_id 是否存在，无 token/会话鉴权；知识库相关接口缺少资源所有权（AuthZ）校验，`user_id` 直接信任前端传入。
3. **FAISS 反序列化风险**：加载 `index.pkl` 使用 `allow_dangerous_deserialization=True`（pickle），加载不可信向量库文件存在 RCE 风险。
4. **同步/异步混用隐患**：部分 repository 函数（如 `list_kbs_from_db`、`kb_exists`、`count_files_from_db` 等）在 `@with_async_session` 下使用同步 `session.query(...)`，在异步引擎下可能报错。
5. **路径穿越防护较基础**：`validate_kb_name` 仅检查 `../`，依赖 `resolve()` 前缀校验。
6. **重复/误导文件**：`create_all_model.py` 与 `create_all_models.py` 内容重复；`server/main.py` 并非应用入口而是 BM25 脚本。
7. **环境耦合**：配置含大量 AutoDL Linux 绝对路径（`/root/autodl-tmp/...`），Windows 本地运行需调整。
8. **外部依赖未完全本地化**：`hwrag` 的 refiner/judger/VLLMGenerator 等仍引用外部 `flashrag` 包。
9. **部分未挂载的能力**：知识库文档管理 API 中已挂载 `list_files`/`upload_docs`/`delete_docs`（含 `user_id` 鉴权 + 异步化）；其余（`update_info`/`recreate_vector_store`/`download_doc`/`update_docs_by_id` 等）仍未挂载，且依赖的 `KBService` 服务层方法仍待异步化（写库类接口还需解决 `user_id` 必填外键的来源）。

---

## 11. 关键文件速查

| 需求 | 入口文件 |
|------|----------|
| 启动服务 | `startup.py` |
| 新增 / 修改 API 路由 | `server/api_router.py` |
| 修改对话逻辑 | `server/chat/chat.py`、`server/chat/knowledge_base_chat.py` |
| 修改检索 / 向量库 | `server/knowledge_base/kb_service/faiss_kb_service.py`、`kb_service/base.py` |
| 修改数据模型 | `server/db/models/*.py` |
| 调整模型 / 端口 / 切分参数 | `configs/model_config.py`、`server_config.py`、`kb_config.py` |
| 调整提示词 | `configs/prompt_config.py` |
| 新增在线模型 worker | `server/model_workers/`（参考 `zhipu.py`） |
| 运行评估 | `hwrag/pipeline/pipeline.py` + `hwrag/config/pipeline_eval.yaml` |
| 文档解析 / 切分 | `document_loaders/`、`text_splitter/` |
| 前端页面（对接全部 API） | `frontend/index.html`（后端启动后访问 `http://<host>:6006/app/`） |
```
