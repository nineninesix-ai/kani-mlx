"""Text-to-speech generation logic"""

import time
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from config import (
    MODEL_NAME, START_OF_HUMAN, END_OF_TEXT, END_OF_HUMAN, END_OF_AI,
    TEMPERATURE, TOP_P, REPETITION_PENALTY, REPETITION_CONTEXT_SIZE, MAX_TOKENS
)


class TTSGenerator:
    def __init__(self):
        self.model, self.tokenizer = load(MODEL_NAME)
        self.sampler = make_sampler(temp=TEMPERATURE, top_p=TOP_P)
        self.logits_processors = make_logits_processors(
            repetition_penalty=REPETITION_PENALTY,
            repetition_context_size=REPETITION_CONTEXT_SIZE
        )

    def prepare_input(self, prompt):
        """Build custom input_ids with special tokens"""
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = mx.array([input_ids])

        start_token = mx.array([[START_OF_HUMAN]], dtype=mx.int64)
        end_tokens = mx.array([[END_OF_TEXT, END_OF_HUMAN]], dtype=mx.int64)
        modified_input_ids = mx.concatenate([start_token, input_ids, end_tokens], axis=1)

        # Flatten to 1D list for generate function
        return modified_input_ids[0].tolist()

    def generate(self, prompt, audio_writer, max_tokens=MAX_TOKENS):
        """Generate speech tokens from text prompt"""
        modified_input_ids = self.prepare_input(prompt)

        point_1 = time.time()

        # Stream tokens from LLM
        generated_text = ""
        all_token_ids = []

        for response in stream_generate(
            self.model,
            self.tokenizer,
            modified_input_ids,
            max_tokens=max_tokens,
            sampler=self.sampler,
            logits_processors=self.logits_processors,
        ):
            generated_text += response.text

            # Use token ID directly from response if available
            if hasattr(response, 'token') and response.token is not None:
                token_id = response.token
                all_token_ids.append(token_id)
                # print(f"[LLM] Token {len(all_token_ids)}: {token_id}")
                audio_writer.add_token(token_id)

                # Stop after END_OF_AI to avoid generating multiple turns
                if token_id == END_OF_AI:
                    print(f"[LLM] END_OF_AI detected, stopping generation")
                    break
            else:
                # Fallback to encoding (shouldn't happen with proper stream_generate)
                print(f"[LLM] Warning: No token ID in response, using text encoding")

        point_2 = time.time()

        print(f"\n[MAIN] Generation complete. Total tokens: {len(all_token_ids)}")
        print(f"[MAIN] Generated text length: {len(generated_text)} chars")

        return {
            'generated_text': generated_text,
            'all_token_ids': all_token_ids,
            'generation_time': point_2 - point_1,
            'point_1': point_1,
            'point_2': point_2
        }
