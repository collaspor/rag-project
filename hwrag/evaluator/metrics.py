import re
import warnings
from collections import Counter

from hwrag.evaluator.utils import normalize_answer


def ensure_answer_list(golden_answers):
    if golden_answers is None:
        return []
    if isinstance(golden_answers, str):
        return [golden_answers]
    return list(golden_answers)


def tokenize_text(text: str):
    normalized = normalize_answer(text)
    if not normalized:
        return []

    # 中文优先尝试分词，分词不可用时再退化到字符级。
    if re.search(r"[\u4e00-\u9fff]", normalized):
        try:
            import jieba

            segmented_tokens = [token.strip() for token in jieba.lcut(normalized) if token.strip()]
            if segmented_tokens:
                return segmented_tokens
        except Exception:
            pass

        return [char for char in normalized if not char.isspace()]

    return normalized.split()


def filter_meaningful_tokens(tokens):
    filtered_tokens = []
    for token in tokens:
        cleaned_token = token.strip()
        if not cleaned_token:
            continue

        if re.fullmatch(r"[a-z0-9]+", cleaned_token):
            filtered_tokens.append(cleaned_token)
            continue

        if re.search(r"[\u4e00-\u9fff]", cleaned_token):
            if len(cleaned_token) >= 2:
                filtered_tokens.append(cleaned_token)
            continue

        if len(cleaned_token) >= 2:
            filtered_tokens.append(cleaned_token)

    return filtered_tokens or tokens


def token_overlap_ratio(source_tokens, target_tokens):
    source_tokens = filter_meaningful_tokens(source_tokens)
    source_counter = Counter(source_tokens)
    target_counter = Counter(target_tokens)
    if not source_counter:
        return 0.0

    overlap = source_counter & target_counter
    overlap_count = sum(overlap.values())
    total_count = sum(source_counter.values())
    if total_count == 0:
        return 0.0
    return overlap_count / total_count


def get_retrieval_match_threshold(config):
    metric_setting = config.get("metric_setting", {})
    return metric_setting.get("retrieval_match_threshold", 0.6)


def calculate_doc_hit(doc_content: str, golden_answers, config):
    normalized_doc = normalize_answer(doc_content)
    if not normalized_doc:
        return False, 0.0

    doc_tokens = tokenize_text(doc_content)
    match_threshold = get_retrieval_match_threshold(config)
    best_score = 0.0

    for answer in ensure_answer_list(golden_answers):
        normalized_answer = normalize_answer(answer)
        if not normalized_answer:
            continue

        if normalized_answer in normalized_doc:
            return True, 1.0

        answer_tokens = tokenize_text(answer)
        overlap_ratio = token_overlap_ratio(answer_tokens, doc_tokens)
        best_score = max(best_score, overlap_ratio)

        if overlap_ratio >= match_threshold:
            return True, overlap_ratio

    return False, best_score


def build_doc_hit_list(doc_list, golden_answers, topk, config):
    hit_list = []
    score_list = []
    doc_contents = [doc["contents"] for doc in doc_list[:topk]]
    for doc in doc_contents:
        hit, score = calculate_doc_hit(doc, golden_answers, config)
        hit_list.append(hit)
        score_list.append(score)

    return hit_list, score_list


class BaseMetric:
    """Base class for all evaluation metrics."""

    metric_name = "base"

    def __init__(self, config):
        self.config = config
        self.dataset_name = config["dataset_name"]

    def calculate_metric(self, data):
        return {}, []


class F1_Score(BaseMetric):
    metric_name = "f1"

    def token_level_scores(self, prediction: str, ground_truths):
        final_metric = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        ground_truths = ensure_answer_list(ground_truths)

        normalized_prediction = normalize_answer(prediction)
        if not normalized_prediction:
            return final_metric

        for ground_truth in ground_truths:
            normalized_ground_truth = normalize_answer(ground_truth)

            if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                continue

            prediction_tokens = tokenize_text(prediction)
            ground_truth_tokens = tokenize_text(ground_truth)
            if not prediction_tokens or not ground_truth_tokens:
                continue

            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = num_same / len(prediction_tokens)
            recall = num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

            final_metric["f1"] = max(f1, final_metric["f1"])
            final_metric["precision"] = max(precision, final_metric["precision"])
            final_metric["recall"] = max(recall, final_metric["recall"])

        return final_metric

    def calculate_metric(self, data):
        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["f1"]
            for pred, golden_answers in zip(data.pred, data.golden_answers)
        ]
        if not metric_score_list:
            return {"f1": 0.0}, []
        f1 = sum(metric_score_list) / len(metric_score_list)
        return {"f1": f1}, metric_score_list


class Recall_Score(F1_Score):
    metric_name = "recall"

    def calculate_metric(self, data):
        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["recall"]
            for pred, golden_answers in zip(data.pred, data.golden_answers)
        ]
        if not metric_score_list:
            return {"recall": 0.0}, []
        recall = sum(metric_score_list) / len(metric_score_list)
        return {"recall": recall}, metric_score_list


class Precision_Score(F1_Score):
    metric_name = "precision"

    def calculate_metric(self, data):
        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["precision"]
            for pred, golden_answers in zip(data.pred, data.golden_answers)
        ]
        if not metric_score_list:
            return {"precision": 0.0}, []
        precision = sum(metric_score_list) / len(metric_score_list)
        return {"precision": precision}, metric_score_list


class ExactMatch(BaseMetric):
    metric_name = "em"

    def __init__(self, config):
        super().__init__(config)
        self.is_regex = self.dataset_name == "curatedtrec"

    def calculate_em(self, prediction: str, golden_answers) -> float:
        golden_answers = ensure_answer_list(golden_answers)
        normalized_prediction = normalize_answer(prediction)
        score = 0.0

        for golden_answer in golden_answers:
            if self.is_regex:
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                if re.fullmatch(golden_answer, normalized_prediction) is not None:
                    score = 1.0
                    break
            else:
                normalized_golden_answer = normalize_answer(golden_answer)
                if normalized_golden_answer == normalized_prediction:
                    score = 1.0
                    break

        return score

    def calculate_metric(self, data):
        metric_score_list = [
            self.calculate_em(pred, golden_answers)
            for pred, golden_answers in zip(data.pred, data.golden_answers)
        ]
        if not metric_score_list:
            return {"em": 0.0}, []
        em_score = sum(metric_score_list) / len(metric_score_list)
        return {"em": em_score}, metric_score_list


class Sub_ExactMatch(BaseMetric):
    metric_name = "sub_em"

    def __init__(self, config):
        super().__init__(config)
        self.is_regex = self.dataset_name == "curatedtrec"

    def calculate_sub_em(self, prediction: str, golden_answers) -> float:
        golden_answers = ensure_answer_list(golden_answers)
        normalized_prediction = normalize_answer(prediction)
        score = 0.0

        for golden_answer in golden_answers:
            if self.is_regex:
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                if re.search(golden_answer, normalized_prediction) is not None:
                    score = 1.0
                    break
            else:
                normalized_golden_answer = normalize_answer(golden_answer)
                if normalized_golden_answer in normalized_prediction:
                    score = 1.0
                    break

        return score

    def calculate_metric(self, data):
        metric_score_list = [
            self.calculate_sub_em(pred, golden_answers)
            for pred, golden_answers in zip(data.pred, data.golden_answers)
        ]
        if not metric_score_list:
            return {"sub_em": 0.0}, []
        sub_em_score = sum(metric_score_list) / len(metric_score_list)
        return {"sub_em": sub_em_score}, metric_score_list


class Retrieval_Recall(BaseMetric):
    metric_name = "retrieval_recall"

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["metric_setting"]["retrieval_recall_topk"]

    def calculate_metric(self, data):
        recall_score_list = []
        for doc_list, golden_answers in zip(data.retrieval_result, data.golden_answers):
            golden_answers = ensure_answer_list(golden_answers)
            if len(doc_list) < self.topk:
                warnings.warn(f"Length of retrieved docs is smaller than topk ({self.topk})")

            hit_list, _ = build_doc_hit_list(doc_list, golden_answers, self.topk, self.config)

            recall_score_list.append(1.0 if any(hit_list) else 0.0)

        if not recall_score_list:
            return {f"retrieval_recall_top{self.topk}": 0.0}, []
        recall_score = sum(recall_score_list) / len(recall_score_list)
        return {f"retrieval_recall_top{self.topk}": recall_score}, recall_score_list


class Retrieval_Precision(BaseMetric):
    metric_name = "retrieval_precision"

    def __init__(self, config):
        super().__init__(config)
        metric_setting = config["metric_setting"]
        self.topk = metric_setting.get("retrieval_precision_topk", metric_setting["retrieval_recall_topk"])

    def calculate_metric(self, data):
        precision_score_list = []
        for doc_list, golden_answers in zip(data.retrieval_result, data.golden_answers):
            golden_answers = ensure_answer_list(golden_answers)
            if len(doc_list) < self.topk:
                warnings.warn(f"Length of retrieved docs is smaller than topk ({self.topk})")

            hit_list, _ = build_doc_hit_list(doc_list, golden_answers, self.topk, self.config)

            if not hit_list:
                precision_score_list.append(0.0)
            else:
                precision_score_list.append(sum(hit_list) / len(hit_list))

        if not precision_score_list:
            return {f"retrieval_precision_top{self.topk}": 0.0}, []
        precision_score = sum(precision_score_list) / len(precision_score_list)
        return {f"retrieval_precision_top{self.topk}": precision_score}, precision_score_list


class RetrievalHitRate(BaseMetric):
    metric_name = "retrieval_hit"
    default_topk = 1

    def __init__(self, config):
        super().__init__(config)
        metric_setting = config.get("metric_setting", {})
        self.topk = metric_setting.get(f"hit_topk_{self.default_topk}", self.default_topk)

    def calculate_metric(self, data):
        hit_score_list = []
        for doc_list, golden_answers in zip(data.retrieval_result, data.golden_answers):
            if len(doc_list) < self.topk:
                warnings.warn(f"Length of retrieved docs is smaller than topk ({self.topk})")

            hit_list, _ = build_doc_hit_list(doc_list, golden_answers, self.topk, self.config)
            hit_score_list.append(1.0 if any(hit_list) else 0.0)

        if not hit_score_list:
            return {f"hit@{self.topk}": 0.0}, []
        score = sum(hit_score_list) / len(hit_score_list)
        return {f"hit@{self.topk}": score}, hit_score_list


class HitAt1(RetrievalHitRate):
    metric_name = "hit@1"
    default_topk = 1


class HitAt3(RetrievalHitRate):
    metric_name = "hit@3"
    default_topk = 3


class HitAt5(RetrievalHitRate):
    metric_name = "hit@5"
    default_topk = 5


class MRR(BaseMetric):
    metric_name = "mrr"

    def __init__(self, config):
        super().__init__(config)
        metric_setting = config.get("metric_setting", {})
        self.topk = metric_setting.get("mrr_topk", config.get("retrieval_topk", 1))

    def calculate_metric(self, data):
        reciprocal_rank_list = []
        for doc_list, golden_answers in zip(data.retrieval_result, data.golden_answers):
            if len(doc_list) < self.topk:
                warnings.warn(f"Length of retrieved docs is smaller than topk ({self.topk})")

            hit_list, _ = build_doc_hit_list(doc_list, golden_answers, self.topk, self.config)
            reciprocal_rank = 0.0
            for rank, hit in enumerate(hit_list, start=1):
                if hit:
                    reciprocal_rank = 1.0 / rank
                    break
            reciprocal_rank_list.append(reciprocal_rank)

        if not reciprocal_rank_list:
            return {"mrr": 0.0}, []
        score = sum(reciprocal_rank_list) / len(reciprocal_rank_list)
        return {"mrr": score}, reciprocal_rank_list


class Rouge_Score(BaseMetric):
    metric_name = "rouge_score"

    def __init__(self, config):
        super().__init__(config)
        from rouge import Rouge

        self.scorer = Rouge()

    def calculate_rouge(self, pred, golden_answers):
        golden_answers = ensure_answer_list(golden_answers)
        output = {}
        for answer in golden_answers:
            scores = self.scorer.get_scores(pred, answer)
            for key in ["rouge-1", "rouge-2", "rouge-l"]:
                output.setdefault(key, [])
                output[key].append(scores[0][key]["f"])

        for key, value in output.items():
            output[key] = max(value)

        return output


class Rouge_1(Rouge_Score):
    metric_name = "rouge-1"

    def calculate_metric(self, data):
        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-1"]
            for pred, golden_answers in zip(data.pred, data.golden_answers)
        ]
        if not metric_score_list:
            return {"rouge-1": 0.0}, []
        score = sum(metric_score_list) / len(metric_score_list)
        return {"rouge-1": score}, metric_score_list


class Rouge_2(Rouge_Score):
    metric_name = "rouge-2"

    def calculate_metric(self, data):
        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-2"]
            for pred, golden_answers in zip(data.pred, data.golden_answers)
        ]
        if not metric_score_list:
            return {"rouge-2": 0.0}, []
        score = sum(metric_score_list) / len(metric_score_list)
        return {"rouge-2": score}, metric_score_list


class Rouge_L(Rouge_Score):
    metric_name = "rouge-l"

    def calculate_metric(self, data):
        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-l"]
            for pred, golden_answers in zip(data.pred, data.golden_answers)
        ]
        if not metric_score_list:
            return {"rouge-l": 0.0}, []
        score = sum(metric_score_list) / len(metric_score_list)
        return {"rouge-l": score}, metric_score_list


class BLEU(BaseMetric):
    metric_name = "bleu"

    def __init__(self, config):
        super().__init__(config)
        from ._bleu import Tokenizer13a

        self.tokenizer = Tokenizer13a()
        self.max_order = config["metric_setting"].get("bleu_max_order", 4)
        self.smooth = config["metric_setting"].get("bleu_smooth", False)

    def calculate_metric(self, data):
        from ._bleu import compute_bleu

        pred_list = [self.tokenizer(pred) for pred in data.pred]
        golden_answers_list = [
            [self.tokenizer(answer) for answer in ensure_answer_list(golden_answers)]
            for golden_answers in data.golden_answers
        ]
        if not pred_list:
            return {"bleu": 0.0}, []

        total_bleu, _, _, _, _, _ = compute_bleu(
            reference_corpus=golden_answers_list,
            translation_corpus=pred_list,
            max_order=self.max_order,
            smooth=self.smooth,
        )

        score_list = []
        for pred, golden_answers in zip(pred_list, golden_answers_list):
            bleu, _, _, _, _, _ = compute_bleu(
                reference_corpus=[golden_answers],
                translation_corpus=[pred],
                max_order=self.max_order,
                smooth=self.smooth,
            )
            score_list.append(bleu)

        return {"bleu": total_bleu}, score_list


class CountToken(BaseMetric):
    metric_name = "input_tokens"

    def __init__(self, config):
        super().__init__(config)
        tokenizer_name = config["metric_setting"].get("tokenizer_name", None)
        is_hf_tokenizer = True
        from flashrag.utils.constants import OPENAI_MODEL_DICT

        if tokenizer_name is None or tokenizer_name in OPENAI_MODEL_DICT:
            import tiktoken

            if tokenizer_name is None:
                tokenizer_name = "gpt-4"
            tokenizer = tiktoken.encoding_for_model(tokenizer_name)
            is_hf_tokenizer = False
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.tokenizer = tokenizer
        self.is_hf_tokenizer = is_hf_tokenizer

    def calculate_metric(self, data):
        input_prompts = data.prompt
        if not input_prompts:
            return {"avg_input_tokens": 0.0}, []

        if self.is_hf_tokenizer:
            token_counts = [len(self.tokenizer.tokenize(text)) for text in input_prompts]
        else:
            token_counts = [len(self.tokenizer.encode(text)) for text in input_prompts]

        avg_tokens = sum(token_counts) / len(token_counts)
        return {"avg_input_tokens": avg_tokens}, token_counts
