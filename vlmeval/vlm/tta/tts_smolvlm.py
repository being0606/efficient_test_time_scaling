import re

import torch
from PIL import Image

from ...smp import *
from ..smolvlm import SmolVLM2


class TTS_SmolVLM2(SmolVLM2):
    """Pure Test-Time Scaling adapter for SmolVLM2.

    Generates N independent samples from the SAME prompt via temperature
    sampling, then aggregates at the answer level. No input augmentation.

    Aggregation methods:
      - answer_level_temperature_majority_vote (Self-Consistency)
      - answer_level_greedy_confidence_scores  (Sample-and-Rank by log-prob)
      - answer_level_greedy_mllm_selector      (Self-Selector via MLLM)
    """

    def __init__(self, model_args, **adapter_args):
        self.model_args = model_args
        self.adapter_args = adapter_args

        for key, value in adapter_args.items():
            setattr(self, key, value)

        self.number_of_versions = getattr(self, "number_of_versions", 4)
        self.token_selection_aggregation_method = getattr(
            self,
            "token_selection_aggregation_method",
            "answer_level_temperature_majority_vote",
        )

        super().__init__(**model_args)

    def _normalize_answer(self, text):
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def _generate_one(self, formatted_message, images, return_confidence=False):
        inputs = self.processor(
            text=formatted_message,
            images=images if len(images) else None,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = dict(self.kwargs)
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = 0.7
        gen_kwargs["top_p"] = 0.95

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            if return_confidence:
                generated = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                sequences = generated.sequences
                transition_scores = self.model.compute_transition_scores(
                    sequences=sequences,
                    scores=generated.scores,
                    normalize_logits=True,
                )
                confidence = transition_scores.sum(dim=1)[0].item()
            else:
                sequences = self.model.generate(**inputs, **gen_kwargs)
                confidence = None

        response = self.processor.batch_decode(
            sequences[:, input_len:],
            skip_special_tokens=True,
        )[0].strip()

        return response, confidence

    def generate_inner_helper(self, formatted_message, images):
        need_confidence = (
            self.token_selection_aggregation_method
            == "answer_level_greedy_confidence_scores"
        )

        responses = []
        confidences = []
        for _ in range(self.number_of_versions):
            response, confidence = self._generate_one(
                formatted_message,
                images,
                return_confidence=need_confidence,
            )
            responses.append(response)
            confidences.append(confidence)

        return responses, confidences

    def answer_level_greedy_majority_vote_generate(self, responses):
        vote_counter = {}
        for idx, response in enumerate(responses):
            key = self._normalize_answer(response)
            if key not in vote_counter:
                vote_counter[key] = {"count": 0, "first_idx": idx}
            vote_counter[key]["count"] += 1

        winner_key = max(
            vote_counter.keys(),
            key=lambda k: (vote_counter[k]["count"], -vote_counter[k]["first_idx"]),
        )
        winner_idx = vote_counter[winner_key]["first_idx"]
        return responses[winner_idx]

    def answer_level_greedy_confidence_scores_generate(self, responses, confidences):
        best_idx = max(range(len(responses)), key=lambda i: confidences[i])
        return responses[best_idx]

    def answer_level_greedy_mllm_selector_generate(self, message, images, responses):
        original_text = " ".join(
            item["value"] for item in message if item.get("type") == "text"
        ).strip()

        candidates = "\n".join(
            f"Answer {i}: {r.strip()}" for i, r in enumerate(responses)
        )
        max_idx = self.number_of_versions - 1

        selector_prompt = "<|im_start|>User:"
        for _ in images:
            selector_prompt += "<image>"
        selector_prompt += (
            f"Question: {original_text}\n"
            f"Different people answered this question in different ways. "
            f"Select the best response from these candidate answers:\n"
            f"{candidates}\n"
            f"Return only the index of the best response as a single integer "
            f"between 0 and {max_idx}."
        )
        selector_prompt += "<end_of_utterance>\nAssistant:"

        inputs = self.processor(
            text=selector_prompt,
            images=images if len(images) else None,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]
        select_kwargs = dict(self.kwargs)
        select_kwargs["max_new_tokens"] = 8
        select_kwargs["do_sample"] = False

        with torch.inference_mode():
            sequences = self.model.generate(**inputs, **select_kwargs)

        decoded = self.processor.batch_decode(
            sequences[:, input_len:],
            skip_special_tokens=True,
        )[0].strip()

        match = re.search(r"\d+", decoded)
        if match is not None:
            idx = int(match.group(0))
            if 0 <= idx < len(responses):
                return responses[idx]

        return self.answer_level_greedy_majority_vote_generate(responses)

    def generate_inner(self, message, dataset=None):
        formatted_message, formatted_images = self.build_prompt_cases(message, dataset)
        images = (
            [formatted_images]
            if isinstance(formatted_images, Image.Image)
            else formatted_images
        )

        responses, confidences = self.generate_inner_helper(formatted_message, images)

        method = self.token_selection_aggregation_method
        if method == "answer_level_greedy_confidence_scores":
            return self.answer_level_greedy_confidence_scores_generate(
                responses, confidences
            )
        elif method == "answer_level_greedy_mllm_selector":
            return self.answer_level_greedy_mllm_selector_generate(
                message, images, responses
            )
        else:  # answer_level_temperature_majority_vote (default)
            return self.answer_level_greedy_majority_vote_generate(responses)
