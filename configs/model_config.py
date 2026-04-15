
MODEL_ROOT_PATH = ""

TEMPERATURE = 0.8

# LLM_MODELS = ["zhipu-api"]

LLM_MODELS = ["chatglm3-6b", "zhipu-api"]
# LLM_MODELS = ["chatglm3-6b"]
MODEL_PATH = {
    # 这里定义 本机服务器上存储的大模型权重存储路径
    "local_model": {
        "chatglm3-6b": "/root/autodl-tmp/data/ZhipuAI/chatglm3-6b",

        # 可扩展其他的开源大模型

    },
    
    # 这里定义 本机服务器上存储的Embedding模型权重存储路径
    "embed_model": {
        "bge-large-zh-v1.5": "/root/autodl-tmp/data/ai-modelscope/bge-large-zh-v1___5",

        # 可扩展其他的Embedding模型
    },
    "reranker": {
        "bge-reranker-large": "/root/autodl-tmp/data/Xorbits/bge-reranker-large",

    }
}

RERANKER_MODEL = "bge-reranker-large"
RERANKER_MAX_LENGTH = 1024
# 是否启用reranker模型
USE_RERANKER = True

# 知识库匹配向量数量
VECTOR_SEARCH_TOP_K = 5
# 知识库匹配的距离阈值，一般取值范围在0-1之间，SCORE越小，距离越小从而相关度越高。
SCORE_THRESHOLD = 1.0
# 如果使用ReRank模型
RERANKER_TOP_K = 3

# 搜索引擎匹配结题数量
SEARCH_ENGINE_TOP_K = 3


ONLINE_LLM_MODEL = {

    # 智谱清言的在线API服务
    "zhipu-api": {
        "api_key": "2c397e8089af28761792c52d334e92c3.fZQHXOprjyAR3Xs7",
        "version": "glm-4",
        "provider": "ChatGLMWorker",
    },

    # OpenAI GPT模型的在线服务
    "openai-api": {
        "model_name": "gpt-4",
        "api_base_url": "https://api.openai.com/v1",
        "api_key": "",
        "openai_proxy": "",
    },

    # 可扩展其他的模型在线模型
    
}



# 选用的 Embedding 名称
EMBEDDING_MODEL = "bge-large-zh-v1.5"

# Embedding 模型运行设备。设为 "auto" 会自动检测(会有警告)，也可手动设定为 "cuda","mps","cpu","xpu" 其中之一。
EMBEDDING_DEVICE = "auto"