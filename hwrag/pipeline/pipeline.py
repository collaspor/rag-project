from hwrag.evaluator import Evaluator
from hwrag.dataset.utils import split_dataset, merge_dataset
from hwrag.utils import get_retriever, get_generator, get_refiner, get_judger
from hwrag.prompt import PromptTemplate


class BasicPipeline:
    """
    基类。一个管道包括了整个 RAG 流程。
    如果想实现一个管道，需要继承这个类。
    """

    def __init__(self, config, prompt_template=None):
        """初始化函数，设置管道配置和默认组件。"""
        self.config = config
        self.device = config["device"]
        self.retriever = None
        self.evaluator = Evaluator(config)
        self.save_retrieval_cache = config["save_retrieval_cache"]

        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template

    def run(self, dataset):
        """执行 RAG 框架的整体推理过程。"""
        pass

    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        """模型完成推理后的评估流程。"""
        if pred_process_fun is not None:
            raw_pred = dataset.pred
            processed_pred = [pred_process_fun(pred) for pred in raw_pred]
            dataset.update_output("raw_pred", raw_pred)
            dataset.update_output("pred", processed_pred)

        if do_eval:
            eval_result = self.evaluator.evaluate(dataset)
            print(eval_result)

        if self.save_retrieval_cache:
            self.retriever._save_cache()

        return dataset


class SequentialPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None):
        """
        检索流程：
            query -> retriever -> generator
        """
        super().__init__(config, prompt_template)
        self.retriever = get_retriever(config)
        self.generator = get_generator(config)

    def naive_run(self, dataset, do_eval=True, pred_process_fun=None):
        input_prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]
        dataset.update_output("prompt", input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_query = dataset.question

        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output("retrieval_result", retrieval_results)

        input_prompts = [
            self.prompt_template.get_string(question=q, retrieval_result=r)
            for q, r in zip(dataset.question, dataset.retrieval_result)
        ]
        dataset.update_output("prompt", input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset


class ConditionalPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None):
        """
        检索流程：
            query -> judger -> sequential pipeline or naive generate
        """
        super().__init__(config, prompt_template)
        self.judger = get_judger(config)

        self.sequential_pipeline = SequentialPipeline(config, prompt_template)
        from flashrag.prompt import PromptTemplate as FlashragPromptTemplate

        self.zero_shot_templete = FlashragPromptTemplate(
            config=config,
            system_prompt=(
                "Answer the question based on your own knowledge. "
                "Only give me the answer and do not output any other words."
            ),
            user_prompt="Question: {question}",
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        judge_result = self.judger.judge(dataset)
        dataset.update_output("judge_result", judge_result)

        pos_dataset, neg_dataset = split_dataset(dataset, judge_result)

        pos_dataset = self.sequential_pipeline.run(pos_dataset, do_eval=False)
        self.sequential_pipeline.prompt_template = self.zero_shot_templete
        neg_dataset = self.sequential_pipeline.naive_run(neg_dataset, do_eval=False)

        dataset = merge_dataset(pos_dataset, neg_dataset, judge_result)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import yaml
    from hwrag.utils import get_dataset
    from hwrag.prompt import PromptTemplate

    def load_runtime_config(config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        if not isinstance(config, dict):
            raise ValueError(f"Config file must contain a top-level mapping: {config_path}")
        return config

    def prepare_runtime_config(config: dict) -> dict:
        if isinstance(config.get("split"), str):
            config["split"] = [config["split"]]

        if not config.get("split"):
            raise ValueError("`split` must contain at least one dataset split.")

        save_dir = config.get("save_dir")
        if save_dir:
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            dataset_name = config.get("dataset_name", "dataset")
            split_name = config["split"][0]
            run_save_dir = os.path.join(save_dir, f"{dataset_name}_{split_name}_{timestamp}")
            os.makedirs(run_save_dir, exist_ok=True)
            config["save_dir"] = run_save_dir

            config_save_path = os.path.join(run_save_dir, "config.yaml")
            with open(config_save_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

        return config

    parser = argparse.ArgumentParser(description="Run the hwrag evaluation pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "pipeline_eval.yaml"),
        help="Path to the pipeline YAML config file.",
    )
    args = parser.parse_args()

    config = prepare_runtime_config(load_runtime_config(args.config))

    all_split = get_dataset(config)
    for split, dataset in all_split.items():
        if dataset is not None and len(dataset) > 0:
            print("ID:" + dataset.id[0])
            print(f"output:{dataset.output[0]}")
            print(f"metadata:{dataset.metadata[0]}")
            print(f"First question in {split} dataset: {dataset.question[0]}")
            print(f"First golden answer in {split} dataset: {dataset.golden_answers[0]}")

    target_split = config["split"][0]
    print(f"Running pipeline on split: {target_split}")
    test_data = all_split.get(target_split)
    if test_data is None:
        raise FileNotFoundError(
            f"Dataset split '{target_split}' was not loaded. "
            f"Expected file: {os.path.join(config['dataset_path'], f'{target_split}.jsonl')}"
        )

    prompt_config = config.get("prompt_template", {})
    prompt_templete = PromptTemplate(
        config,
        system_prompt=prompt_config.get(
            "system_prompt",
            "根据给定文档回答问题。只给出答案，不要输出其他无关内容。\n下面是给定的参考文档：\n{reference}",
        ),
        user_prompt=prompt_config.get(
            "user_prompt",
            "问题: {question}\n答案:",
        ),
    )

    pipeline = SequentialPipeline(config, prompt_template=prompt_templete)
    output_dataset = pipeline.run(test_data, do_eval=True)

    print("---generation output---")
    for single_reponse in output_dataset.pred:
        print(single_reponse)
