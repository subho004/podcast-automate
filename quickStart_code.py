#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import soundfile as sf
import numpy as np
import time

CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7

SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
BOS_ID = 128000
TEXT_EOT_ID = 128009


def build_prompt(tokenizer, description: str, text: str) -> str:
    """Build formatted prompt for Maya1."""
    soh_token = tokenizer.decode([SOH_ID])
    eoh_token = tokenizer.decode([EOH_ID])
    soa_token = tokenizer.decode([SOA_ID])
    sos_token = tokenizer.decode([CODE_START_TOKEN_ID])
    eot_token = tokenizer.decode([TEXT_EOT_ID])
    bos_token = tokenizer.bos_token
    
    # Place the spoken text first, then mark end-of-text, and supply the
    # `description` after the end marker so it is available as context
    # but not part of the spoken tokens the model should emit.
    formatted_text = f'{text}'

    prompt = (
        soh_token + bos_token + formatted_text + eot_token +
        eoh_token + f'<description="{description}">' + soa_token + sos_token
    )
    
    return prompt


def extract_snac_codes(token_ids: list) -> list:
    """Extract SNAC codes from generated tokens."""
    try:
        eos_idx = token_ids.index(CODE_END_TOKEN_ID)
    except ValueError:
        eos_idx = len(token_ids)
    
    snac_codes = [
        token_id for token_id in token_ids[:eos_idx]
        if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID
    ]
    
    return snac_codes


def unpack_snac_from_7(snac_tokens: list) -> list:
    """Unpack 7-token SNAC frames to 3 hierarchical levels."""
    if snac_tokens and snac_tokens[-1] == CODE_END_TOKEN_ID:
        snac_tokens = snac_tokens[:-1]
    
    frames = len(snac_tokens) // SNAC_TOKENS_PER_FRAME
    snac_tokens = snac_tokens[:frames * SNAC_TOKENS_PER_FRAME]
    
    if frames == 0:
        return [[], [], []]
    
    l1, l2, l3 = [], [], []
    
    for i in range(frames):
        slots = snac_tokens[i*7:(i+1)*7]
        l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
        l2.extend([
            (slots[1] - CODE_TOKEN_OFFSET) % 4096,
            (slots[4] - CODE_TOKEN_OFFSET) % 4096,
        ])
        l3.extend([
            (slots[2] - CODE_TOKEN_OFFSET) % 4096,
            (slots[3] - CODE_TOKEN_OFFSET) % 4096,
            (slots[5] - CODE_TOKEN_OFFSET) % 4096,
            (slots[6] - CODE_TOKEN_OFFSET) % 4096,
        ])
    
    return [l1, l2, l3]


def main():
    # simple synthesize function loads models, generates, decodes, and writes output
    def synthesize(text: str, description: str, output_file: str = "output.wav", model_name: str = "maya-research/maya1") -> dict:
        total_start = time.perf_counter()

        # Load model
        print("\n[1/3] Loading Maya1 model...")
        model_start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_elapsed = time.perf_counter() - model_start
        print(f"Model loaded: {len(tokenizer)} tokens in vocabulary")
        print(f"Model load time: {model_elapsed:.2f}s")

        # Load SNAC
        print("\n[2/3] Loading SNAC audio decoder...")
        snac_start = time.perf_counter()
        snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        if torch.cuda.is_available():
            snac_model = snac_model.to("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            snac_model = snac_model.to("mps")
        snac_elapsed = time.perf_counter() - snac_start
        print("SNAC decoder loaded")
        print(f"SNAC load time: {snac_elapsed:.2f}s")

        # Build prompt
        prompt = build_prompt(tokenizer, description, text)
        print("\n[3/3] Generating speech...")
        print(f"Description: {description}")
        print(f"Text: {text}")
        print(f"\nPrompt preview (first 200 chars):")
        print(f"   {repr(prompt[:200])}")
        print(f"   Prompt length: {len(prompt)} chars")

        gen_start = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt")
        print(f"   Input token count: {inputs['input_ids'].shape[1]} tokens")
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                min_new_tokens=28,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                eos_token_id=CODE_END_TOKEN_ID,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_elapsed = time.perf_counter() - gen_start
        print(f"Generation time: {gen_elapsed:.2f}s")

        generated_ids = outputs[0, inputs['input_ids'].shape[1]:].tolist()
        print(f"Generated {len(generated_ids)} tokens")

        snac_tokens = extract_snac_codes(generated_ids)
        print(f"Extracted {len(snac_tokens)} SNAC tokens")

        if len(snac_tokens) < 7:
            raise RuntimeError("Not enough SNAC tokens generated")

        levels = unpack_snac_from_7(snac_tokens)
        frames = len(levels[0])
        print(f"Unpacked to {frames} frames")

        try:
            snac_device = next(snac_model.parameters()).device
        except StopIteration:
            snac_device = torch.device("cpu")

        codes_tensor = [
            torch.tensor(level, dtype=torch.long, device=snac_device).unsqueeze(0)
            for level in levels
        ]

        print("\n[4/4] Decoding to audio...")
        decode_start = time.perf_counter()
        with torch.inference_mode():
            z_q = snac_model.quantizer.from_codes(codes_tensor)
            audio = snac_model.decoder(z_q)[0, 0].cpu().numpy()
        decode_elapsed = time.perf_counter() - decode_start
        print(f"Decoding time: {decode_elapsed:.2f}s")

        if len(audio) > 2048:
            audio = audio[2048:]

        sf.write(output_file, audio, 24000)
        duration_sec = len(audio) / 24000
        total_elapsed = time.perf_counter() - total_start
        print(f"Audio generated: {len(audio)} samples ({duration_sec:.2f}s)")
        print(f"Voice generated successfully! -> {output_file}")
        print(f"Total run time: {total_elapsed:.2f}s")

        return {
            "output_file": output_file,
            "duration": duration_sec,
            "timings": {
                "model_load": model_elapsed,
                "snac_load": snac_elapsed,
                "generation": gen_elapsed,
                "decoding": decode_elapsed,
                "total": total_elapsed,
            },
        }

    # default CLI behavior: synthesize the demo text
    synthesize(
        text="Hello! This is Maya1 <laugh_harder> the best open source voice AI model with emotions.",
        description="Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing.",
        output_file="output.wav",
    )


if __name__ == "__main__":
    main()
