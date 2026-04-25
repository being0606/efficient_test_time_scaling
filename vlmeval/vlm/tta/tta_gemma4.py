import re

import torch
from PIL import Image

from ...smp import *
from ..gemma4 import Gemma4


class TTS_Gemma4(Gemma4):
    """Pure Test-Time Scaling adapter for Gemma4.

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

    def _build_prompt(self, message, images):
        content = []
        image_idx = 0
        for item in message:
            if item.get("type") == "text":
                content.append({"type": "text", "text": item["value"]})
            elif item.get("type") == "image":
                content.append({"type": "image", "image": images[image_idx]})
                image_idx += 1

        messages = [{"role": "user", "content": content}]
        try:
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _generate_one(self, prompt, images, return_confidence=False):
        inputs = self.processor(
            text=prompt,
            images=images if len(images) else None,
            return_tensors="pt",
        )

        device = getattr(self.model, "device", None)
        if device is None:
            device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        gen_kwargs = dict(self.generate_kwargs)
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
            clean_up_tokenization_spaces=False,
        )[0].strip()

        if hasattr(self.processor, "parse_response"):
            try:
                parsed = self.processor.parse_response(response)
                if isinstance(parsed, str) and len(parsed):
                    response = parsed
            except Exception:
                pass

        return response, confidence

    def generate_inner_helper(self, message, dataset=None):
        """Generate N responses from the SAME prompt via temperature sampling."""
        original_images = [
            Image.open(item["value"]).convert("RGB")
            for item in message
            if item.get("type") == "image"
        ]
        prompt = self._build_prompt(message, original_images)

        need_confidence = (
            self.token_selection_aggregation_method
            == "answer_level_greedy_confidence_scores"
        )

        responses = []
        confidences = []
        for _ in range(self.number_of_versions):
            response, confidence = self._generate_one(
                prompt,
                original_images,
                return_confidence=need_confidence,
            )
            responses.append(response)
            confidences.append(confidence)

        return responses, confidences

    def answer_level_greedy_majority_vote_generate(self, responses):
        """Select best response via majority voting."""
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
        """Select best response via confidence scores (sum of log-probs)."""
        best_idx = max(range(len(responses)), key=lambda i: confidences[i])
        return responses[best_idx]

    def answer_level_greedy_mllm_selector_generate(self, message, responses):
        """Ask the model itself to pick the best candidate answer by index."""
        original_text = " ".join(
            item["value"] for item in message if item.get("type") == "text"
        ).strip()
        images = [
            Image.open(item["value"]).convert("RGB")
            for item in message
            if item.get("type") == "image"
        ]

        candidates = "\n".join(
            f"Answer {i}: {r.strip()}" for i, r in enumerate(responses)
        )
        max_idx = self.number_of_versions - 1
        selector_text = (
            f"Question: {original_text}\n"
            f"Different people answered this question in different ways. "
            f"Select the best response from these candidate answers:\n"
            f"{candidates}\n"
            f"Return only the index of the best response as a single integer "
            f"between 0 and {max_idx}."
        )

        selector_message = [{"role": "user", "content": []}]
        for img in images:
            selector_message[0]["content"].append({"type": "image", "image": img})
        selector_message[0]["content"].append({"type": "text", "text": selector_text})

        try:
            prompt = self.processor.apply_chat_template(
                selector_message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            prompt = self.processor.apply_chat_template(
                selector_message,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.processor(
            text=prompt,
            images=images if len(images) else None,
            return_tensors="pt",
        )
        device = getattr(self.model, "device", None)
        if device is None:
            device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        input_len = inputs["input_ids"].shape[-1]
        select_kwargs = dict(self.generate_kwargs)
        select_kwargs["max_new_tokens"] = 8
        select_kwargs["do_sample"] = False

        with torch.inference_mode():
            sequences = self.model.generate(**inputs, **select_kwargs)

        decoded = self.processor.batch_decode(
            sequences[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        match = re.search(r"\d+", decoded)
        if match is not None:
            idx = int(match.group(0))
            if 0 <= idx < len(responses):
                return responses[idx]

        return self.answer_level_greedy_majority_vote_generate(responses)

    def generate_inner(self, message, dataset=None):
        """Main generation pipeline with aggregation method selection."""
        responses, confidences = self.generate_inner_helper(message, dataset)

        method = self.token_selection_aggregation_method
        if method == "answer_level_greedy_confidence_scores":
            return self.answer_level_greedy_confidence_scores_generate(responses, confidences)
        elif method == "answer_level_greedy_mllm_selector":
            return self.answer_level_greedy_mllm_selector_generate(message, responses)
        else:  # answer_level_temperature_majority_vote (default)
            return self.answer_level_greedy_majority_vote_generate(responses)
