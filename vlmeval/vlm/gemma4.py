from PIL import Image
import torch

from .base import BaseModel
from ..smp import *


class Gemma4(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='google/gemma-4-31B-it', **kwargs):
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
        except Exception as e:
            logging.critical('Please install the latest version transformers.')
            raise e

        self.model_path = model_path
        self.enable_thinking = kwargs.pop('enable_thinking', False)
        self.generate_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
        )
        self.generate_kwargs.update(kwargs)

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype='auto',
            device_map='auto',
            trust_remote_code=True,
        ).eval()

    def _build_messages(self, message):
        content = []
        for item in message:
            if item['type'] == 'image':
                content.append({'type': 'image', 'image': item['value']})
            elif item['type'] == 'text':
                content.append({'type': 'text', 'text': item['value']})
            else:
                raise ValueError(f'Unsupported content type: {item["type"]}')

        return [{'role': 'user', 'content': content}]

    def generate_inner(self, message, dataset=None):
        messages = self._build_messages(message)

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

        images = [Image.open(item['value']).convert('RGB') for item in message if item['type'] == 'image']
        inputs = self.processor(
            text=prompt,
            images=images if len(images) else None,
            return_tensors='pt',
        )

        device = getattr(self.model, 'device', None)
        if device is None:
            device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        input_len = inputs['input_ids'].shape[-1]
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self.generate_kwargs)

        response = self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        if hasattr(self.processor, 'parse_response'):
            try:
                parsed = self.processor.parse_response(response)
                if isinstance(parsed, str) and len(parsed):
                    response = parsed
            except Exception:
                pass

        return response.strip()