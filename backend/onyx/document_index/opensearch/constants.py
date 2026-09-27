# Default value for the maximum number of tokens a chunk can hold, if none is
# specified when creating an index.
import os
from enum import Enum

DEFAULT_MAX_CHUNK_SIZE = 512


# By default OpenSearch will only return a maximum of this many results in a
# given search. This value is configurable in the index settings.
DEFAULT_OPENSEARCH_MAX_RESULT_WINDOW = 10_000


# For documents which do not have a value for LAST_UPDATED_FIELD_NAME, we assume
# that the document was last updated this many days ago for the purpose of time
# cutoff filtering during retrieval.
ASSUMED_DOCUMENT_AGE_DAYS = 90


# Size of the dynamic list used to consider elements during kNN graph creation.
# Higher values improve search quality but increase indexing time. Values
# typically range between 100 - 512.
EF_CONSTRUCTION = 256
# Number of bi-directional links per element. Higher values improve search
# quality but increase memory footprint. Values typically range between 12 - 48.
M = 32  # Set relatively high for better accuracy.

# 混合检索会合并各路分数并重新排序，因此候选数量应多于最终返回数量。
# 候选越多，越有机会召回相关内容并参与混合评分，但每次查询的计算开销也越大。
# 例如，最终需要 10 个结果时，若关键词和向量查询各只取 10 个候选，
# 候选重叠不足就可能缺少某一路的得分，甚至漏掉综合评分本应靠前的文档。
# 若各取 1000 个候选，目标文档被召回并获得各路评分的机会更大。
# 未进入候选集的文档无法通过后续重新排序找回。
# 当前默认值为 500；最初设为 750 时查询性能不佳，后来从 100 提升至 500 以改善召回。
DEFAULT_NUM_HYBRID_SUBQUERY_CANDIDATES = int(
    os.environ.get("DEFAULT_NUM_HYBRID_SUBQUERY_CANDIDATES", 500)
)

# Number of vectors to examine to decide the top k neighbors for the HNSW
# method.
# NOTE: "When creating a search query, you must specify k. If you provide both k
# and ef_search, then the larger value is passed to the engine. If ef_search is
# larger than k, you can provide the size parameter to limit the final number of
# results to k." from
# https://docs.opensearch.org/latest/query-dsl/specialized/k-nn/index/#ef_search
EF_SEARCH = DEFAULT_NUM_HYBRID_SUBQUERY_CANDIDATES


class OpenSearchAuthMethod(str, Enum):
    """Authentication method for connecting to OpenSearch.

    BASIC uses HTTP basic auth (username/password); the only option for
    self-hosted / docker-compose OpenSearch. IAM uses AWS SigV4 request signing
    and is only valid against an AWS managed domain whose fine-grained access
    control master is an IAM ARN.
    """

    BASIC = "basic"
    IAM = "iam"


class OpenSearchSearchType(str, Enum):
    """Search type label used for Prometheus metrics."""

    HYBRID = "hybrid"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    RANDOM = "random"
    DOC_ID_RETRIEVAL = "doc_id_retrieval"
    UNKNOWN = "unknown"


class HybridSearchSubqueryConfiguration(Enum):
    TITLE_VECTOR_CONTENT_VECTOR_TITLE_CONTENT_COMBINED_KEYWORD = 1
    # Current default.
    CONTENT_VECTOR_TITLE_CONTENT_COMBINED_KEYWORD = 2


# 从环境变量读取混合检索组合，转换为整数并校验对应的枚举值。
# 未设置时，默认使用“正文向量 + 标题正文关键词”两路检索。
# 已设置但无法转为整数或不属于有效枚举值时，抛出异常并阻止应用启动。
HYBRID_SEARCH_SUBQUERY_CONFIGURATION: HybridSearchSubqueryConfiguration = (
    HybridSearchSubqueryConfiguration(
        int(os.environ["HYBRID_SEARCH_SUBQUERY_CONFIGURATION"])
    )
    if os.environ.get("HYBRID_SEARCH_SUBQUERY_CONFIGURATION", None) is not None
    else HybridSearchSubqueryConfiguration.CONTENT_VECTOR_TITLE_CONTENT_COMBINED_KEYWORD
)


class HybridSearchNormalizationPipeline(Enum):
    # Current default.
    MIN_MAX = 1
    # NOTE: Using z-score normalization is better for hybrid search from a
    # theoretical standpoint. Empirically on a small dataset of up to 10K docs,
    # it's not very different. Likely more impactful at scale.
    # https://opensearch.org/blog/introducing-the-z-score-normalization-technique-for-hybrid-search/
    ZSCORE = 2


# Will raise and block application start if HYBRID_SEARCH_NORMALIZATION_PIPELINE
# is set but not a valid value. If not set, defaults to MIN_MAX.
HYBRID_SEARCH_NORMALIZATION_PIPELINE: HybridSearchNormalizationPipeline = (
    HybridSearchNormalizationPipeline(
        int(os.environ["HYBRID_SEARCH_NORMALIZATION_PIPELINE"])
    )
    if os.environ.get("HYBRID_SEARCH_NORMALIZATION_PIPELINE", None) is not None
    else HybridSearchNormalizationPipeline.MIN_MAX
)
