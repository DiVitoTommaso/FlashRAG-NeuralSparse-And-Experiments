import ast

from bert_score import score

from flashrag.evaluator import ExactMatch
from . import *
from .modules_logger import *


import numpy as np
from datasets import Dataset as HFDataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
)
from ragas import evaluate as ragas_evaluate
from rouge_score import rouge_scorer
import evaluate


class EvaluationModule:
    def __init__(self, pipeline):
        self.pipeline = pipeline


    def fix_evaluate(self, file):
        with open(file, 'r') as f, open(file + 'Ext', 'w') as w:
            for line in f.readlines():
                json = ast.literal_eval(line)

                out = json['output']
                self.pipeline.dataset['pred'] = []
                for query in out:
                    pred = query['generated_original']
                    self.pipeline.dataset['pred'].append(pred
                                                         .split('\n')[0]
                                                         .split('Question:')[0]
                                                         .split('Final answer:')[0]
                                                         .replace("`", "")
                                                         .strip()
                                                         )

                json['fixed_predictions'] = {}
                json['fixed_predictions'] = json['fixed_predictions'] | self.eval_advanced()
                json['fixed_predictions'] = json['fixed_predictions'] | self.eval_standard()
                w.write(json.dumps(json))

    @staticmethod
    def normalize(text_list):
        """
        Normalize a list of text strings by removing newlines, extra spaces,
        and replacing empty strings with a placeholder.
        """
        import re
        new_text = []
        for text in text_list:
            text = text.replace('\n', ' ').replace('\r', ' ')
            text = re.sub(r'[ \t]+', ' ', text)
            text = text.strip()

            if text:
                new_text.append(text)
            else:
                new_text.append('[No answer]')
        return new_text

    def retrieval_evaluate(self):
        float_scores = []
        exact_scores = 0
        for i in range(len(self.pipeline.dataset.queries)):
            answer = self.pipeline.dataset.answers[i]
            docs = self.pipeline.dataset.retrieval[i]

            tokens = answer.split(" ")
            docs = " ".join(docs)

            c = 0
            for token in tokens:
                c += int(token in docs)

            float_scores.append(c / len(tokens))    #Percentage of tokens of answer in documents
            exact_scores += int(c == len(tokens))

        return np.mean(float_scores), exact_scores / len(float_scores)


    def slide_merge(self, generated, queries, answers, eval_on='em'):

        best_generated = []
        for i in range(len(generated)):
            scores = []
            for j in range(len(generated[i])):
                if eval_on == 'em':
                    em = ExactMatch(self.pipeline.config)
                    evaluation = {'em' : em.calculate_em(generated[i][j], answers[i])}

                if eval_on.lower().split('.')[0] in ['bertscore', 'ragas', 'meteor', 'rouge', 'diversity']:
                    evaluation = self.eval_advanced(predictions=generated[i][j:j+1], queries=queries[i:i+1],
                                                    answers=answers[i:i+1], metrics=[eval_on])

                if '.' in eval_on:
                    scores.append(evaluation[eval_on.split('.')[0]][eval_on.split('.')[1]])
                else:
                    scores.append(evaluation[eval_on])

            best_generated.append(generated[i][np.argmax(scores)])

        return best_generated

    def eval_standard(self, dataset=None):
        """
        Run the standard evaluation on the pipeline's dataset,
        capturing printed output and returning it as a dictionary.
        """
        buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = buffer

        try:
            self.pipeline.evaluate(dataset or self.pipeline.dataset.all, do_eval=True)
        finally:
            sys.stdout = original_stdout

        output_str = buffer.getvalue()
        logger.info(f"Standard evaluation output:\n{output_str}")

        # Convert the printed string output to a Python dict safely
        return ast.literal_eval(output_str)

    def eval_advanced(self, predictions=None, queries=None, answers=None, metrics=None) -> dict:
        predictions = predictions or self.pipeline.dataset.predictions
        queries = queries or self.pipeline.dataset.queries
        references = answers or self.pipeline.dataset.answers  # list[list[str]]
        retrieval = [[document['contents'] for document in documents] for documents in self.pipeline.dataset.retrieval]
        metrics = metrics or self.pipeline.config['advanced_metrics']

        final_scores = {}

        # ROUGE
        if 'rouge' in metrics:
            rouge_scores = compute_rouge(self.pipeline)
            final_scores.update(rouge_scores)
            logger.debug("Computed ROUGE scores")

        # BERTScore
        if 'bertscore' in metrics:
            bert_scores = compute_bertscore(self.pipeline)
            final_scores.update(bert_scores)
            logger.debug("Computed BERTScore")

        # RAGAS
        if 'ragas' in metrics:
            ragas_scores = compute_ragas(predictions, queries, references, retrieval)
            final_scores.update(ragas_scores)
            logger.debug("Computed RAGAS scores")

        # METEOR
        if 'meteor' in metrics:
            meteor_scores = compute_meteor(predictions, references)
            final_scores.update(meteor_scores)
            logger.debug("Computed METEOR scores")

        # DIVERSITY
        if 'diversity' in metrics:
            diversity_scores = compute_diversity(predictions)
            final_scores.update(diversity_scores)
            logger.debug("Computed Diversity metrics")

        logger.debug(f"Final evaluation scores: {final_scores}")
        return final_scores


def compute_meteor(predictions, references):
    meteor = evaluate.load("meteor")
    meteor_scores = []

    for pred, refs in zip(predictions, references):
        best_score = 0.0
        for ref in refs:
            score = meteor.compute(predictions=[pred], references=[ref])['meteor']
            best_score = max(best_score, score)
        meteor_scores.append(best_score)

    return {'meteor': np.mean(meteor_scores)}


def compute_diversity(predictions):
    def distinct_n(preds, n=2):
        all_ngrams = set()
        total = 0
        for pred in preds:
            tokens = pred.split()
            ngrams = list(zip(*[tokens[i:] for i in range(n)]))
            all_ngrams.update(ngrams)
            total += len(ngrams)
        return len(all_ngrams) / total if total > 0 else 0

    return {
        "distinct_1": distinct_n(predictions, 1),
        "distinct_2": distinct_n(predictions, 2)
    }


def compute_ragas(preds, questions, golden_answers, contexts):
    """
    Compute RAGAS metrics when there are multiple ground truth answers
    for each question and a single prediction.

    For each (prediction, references), we evaluate against all references
    and select the best match per metric.

    Args:
        preds: List[str], predicted answers.
        questions: List[str], user questions.
        golden_answers: List[List[str]], multiple references per question.
        contexts: List[List[dict]], retrieved contexts for each question.

    Returns:
        Dict[str, float]: Aggregated RAGAS scores.
    """
    metrics = [answer_relevancy, faithfulness, context_precision]
    scores_per_metric = {m.name: [] for m in metrics}

    for pred, q, refs, ctxs in zip(preds, questions, golden_answers, contexts):
        best_scores = {m.name: 0.0 for m in metrics}

        for ref in refs:
            data = [{
                "question": q,
                "answer": pred,
                "ground_truth": ref,
                "contexts": ctxs or [],
            }]
            ragas_dataset = HFDataset.from_list(data)

            try:
                results = ragas_evaluate(ragas_dataset, metrics=metrics)
                for metric in metrics:
                    score = results[metric.name]
                    best_scores[metric.name] = max(best_scores[metric.name], score)
            except Exception as e:
                # If RAGAS fails, skip this reference and log
                print(f"RAGAS failed on (Q: {q}) with ref '{ref}': {e}")
                continue

        for k in best_scores:
            scores_per_metric[k].append(best_scores[k])

    # Compute mean score for each metric
    return {k: float(np.mean(v)) if v else 0.0 for k, v in scores_per_metric.items()}


def compute_rouge(pipeline):
    """
    Compute ROUGE scores (rouge1, rouge2, rougeL) between predicted answers
    and golden answers in the pipeline's dataset.

    Args:
        pipeline: The evaluation pipeline containing dataset with predictions and golden answers.

    Returns:
        Dictionary with average ROUGE scores (precision, recall, fmeasure).
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def best_match_scores(cand, refs):
        """
        Find best ROUGE score among multiple references for a candidate.
        """
        best_score = None
        for ref in refs:
            score = scorer.score(ref, cand)
            if best_score is None or sum(v.fmeasure for v in score.values()) > sum(
                    v.fmeasure for v in best_score.values()):
                best_score = score
        return best_score

    per_sentence_scores = [
        best_match_scores(cand, refs)
        for cand, refs in zip(pipeline.dataset.predictions, pipeline.dataset.answers)
    ]

    def avg_score(metric):
        """
        Compute average precision, recall, and fmeasure for given ROUGE metric.
        """
        return {
            "precision": sum(s[metric].precision for s in per_sentence_scores) / len(per_sentence_scores),
            "recall": sum(s[metric].recall for s in per_sentence_scores) / len(per_sentence_scores),
            "fmeasure": sum(s[metric].fmeasure for s in per_sentence_scores) / len(per_sentence_scores),
        }

    rouge_scores = {metric: avg_score(metric) for metric in ["rouge1", "rouge2", "rougeL"]}
    del scorer
    return rouge_scores


def compute_bertscore(pipeline):
    """
    Compute BERTScore using batch evaluation for better performance.
    For each prediction, the best score among its references is used.
    """
    import evaluate
    bertscore = evaluate.load("bertscore")

    predictions = []
    references = []
    mapping = []  # (pred_index, ref_index) per tenere traccia dell'associazione

    # Flatten all (pred, ref) pairs
    for i, (pred, refs) in enumerate(zip(pipeline.dataset.predictions, pipeline.dataset.answers)):
        for j, ref in enumerate(refs):
            predictions.append(pred)
            references.append(ref)
            mapping.append(i)

    # Compute all scores in batch
    results = bertscore.compute(predictions=predictions, references=references, lang='en')

    # Organizza i punteggi per prediction
    best_scores = {}
    for i, pred_idx in enumerate(mapping):
        score_tuple = (results["precision"][i], results["recall"][i], results["f1"][i])
        if pred_idx not in best_scores or score_tuple[2] > best_scores[pred_idx][2]:  # Use best F1
            best_scores[pred_idx] = score_tuple

    # Calcola medie
    num_examples = len(pipeline.dataset.predictions)
    precision = sum(score[0] for score in best_scores.values()) / num_examples
    recall = sum(score[1] for score in best_scores.values()) / num_examples
    f1 = sum(score[2] for score in best_scores.values()) / num_examples

    del bertscore
    return {
        "bertscore_precision": precision,
        "bertscore_recall": recall,
        "bertscore_f1": f1
    }
