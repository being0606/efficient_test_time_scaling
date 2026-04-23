import copy
import re

import torch
from PIL import Image

from ...smp import *
from ..gemma4 import Gemma4
from .image_augment import ImageAugment
from .text_augment import TextAugment


class _IdentityTextAugment:
    def __init__(self, n_augmentations):
        self.n_augmentations = max(1, n_augmentations)

    def __call__(self, text_prompt):
        return [text_prompt] * self.n_augmentations


class TTAugAdapter_Gemma4(Gemma4):
    def __init__(self, model_args, text_aug_args, image_aug_args, **adapter_args):
        self.model_args = model_args
        self.adapter_args = adapter_args

        for key, value in adapter_args.items():
            setattr(self, key, value)

        self.number_of_versions = getattr(self, "number_of_versions", 4)
        self.token_selection_aggregation_method = getattr(
            self,
            "token_selection_aggregation_method",
            "answer_level_greedy_majority_vote",
        )

        super().__init__(**model_args)

        try:
            self.text_augment = TextAugment(
                n_augmentations=self.number_of_versions,
                **text_aug_args,
            )
        except Exception as e:
            print(
                "TextAugment init failed. Falling back to identity text augmentation.",
                e,
            )
            self.text_augment = _IdentityTextAugment(self.number_of_versions)

        self.image_augment = ImageAugment(
            n_augmentations=self.number_of_versions,
            **image_aug_args,
        )

    def _normalize_answer(self, text):
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def create_message_versions(self, message):
        versions = [copy.deepcopy(message) for _ in range(self.number_of_versions)]

        def split_options(text):
            text = text.strip()
            if not text or len(text.split()) <= 2:
                return None, None
            if "Options:" in text:
                main, opts = text.split("Options:", 1)
                return main.strip(), "Options:" + opts
            return text, ""

        for msg_idx, msg in enumerate(message):
            if msg.get("type") == "text":
                base, opts = split_options(msg["value"])
                if base is None:
                    continue
                paraphrases = self.text_augment(base)
                for ver_idx in range(self.number_of_versions):
                    pv = paraphrases[ver_idx] if ver_idx < len(paraphrases) else base
                    versions[ver_idx][msg_idx]["value"] = pv + opts

        return versions

    def _build_prompt_and_images(self, message_version, images_for_version):
        content = []
        image_idx = 0

        for item in message_version:
            if item.get("type") == "text":
                content.append({"type": "text", "text": item["value"]})
            elif item.get("type") == "image":
                content.append({"type": "image", "image": images_for_version[image_idx]})
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

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            if return_confidence:
                generated = self.model.generate(
                    **inputs,
                    **self.generate_kwargs,
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
                sequences = self.model.generate(**inputs, **self.generate_kwargs)
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
        """Generate responses for all augmented versions."""
        message_versions = self.create_message_versions(message)

        original_images = [
            Image.open(item["value"]).convert("RGB")
            for item in message
            if item.get("type") == "image"
        ]
        augmented_images, _ = self.image_augment(original_images)

        responses = []
        confidences = []

        need_confidence = (
            self.token_selection_aggregation_method
            == "answer_level_greedy_confidence_scores"
        )

        for ver_idx in range(self.number_of_versions):
            prompt = self._build_prompt_and_images(
                message_versions[ver_idx],
                augmented_images[ver_idx],
            )
            response, confidence = self._generate_one(
                prompt,
                augmented_images[ver_idx],
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
        """Select best response via confidence scores."""
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
        else:  # Default to majority vote
            return self.answer_level_greedy_majority_vote_generate(responses)