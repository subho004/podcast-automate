import os
import time
from typing import Dict
from pathlib import Path

import torch

from quickStart_code import (
    build_prompt,
    extract_snac_codes,
    unpack_snac_from_7,
    CODE_END_TOKEN_ID,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import soundfile as sf


class PersistentAudioGenerator:
    """Loads Maya1 and SNAC once and exposes `synthesize_scene`.

    This class keeps models resident in memory to speed up multiple syntheses.
    """
    def __init__(self, out_dir: str = "outputs", model_name: str = "maya-research/maya1", dtype=torch.bfloat16):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.dtype = dtype

        print("Loading synthesis models (this may take a while)...")
        start = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=self.dtype, device_map="auto", trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        if torch.cuda.is_available():
            self.snac_model = self.snac_model.to("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self.snac_model = self.snac_model.to("mps")

        elapsed = time.perf_counter() - start
        print(f"Models loaded in {elapsed:.2f}s")

    def synthesize_scene(self, scene_text: str, voice_description: str, episode_id: str, scene_idx: int) -> Dict:
        ep_dir = self.out_dir / episode_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        out_file = ep_dir / f"scene_{scene_idx:02d}.wav"

        prompt = build_prompt(self.tokenizer, voice_description, scene_text)

        # generation
        gen_start = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                min_new_tokens=28,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                eos_token_id=CODE_END_TOKEN_ID,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_elapsed = time.perf_counter() - gen_start

        generated_ids = outputs[0, inputs["input_ids"].shape[1] :].tolist()
        snac_tokens = extract_snac_codes(generated_ids)

        if len(snac_tokens) < 7:
            raise RuntimeError("Not enough SNAC tokens generated")

        levels = unpack_snac_from_7(snac_tokens)

        try:
            snac_device = next(self.snac_model.parameters()).device
        except StopIteration:
            snac_device = torch.device("cpu")

        codes_tensor = [
            torch.tensor(level, dtype=torch.long, device=snac_device).unsqueeze(0)
            for level in levels
        ]

        decode_start = time.perf_counter()
        with torch.inference_mode():
            z_q = self.snac_model.quantizer.from_codes(codes_tensor)
            audio = self.snac_model.decoder(z_q)[0, 0].cpu().numpy()
        decode_elapsed = time.perf_counter() - decode_start

        if len(audio) > 2048:
            audio = audio[2048:]

        sf.write(str(out_file), audio, 24000)
        duration_sec = len(audio) / 24000
        total_elapsed = gen_elapsed + decode_elapsed

        return {
            "audio": str(out_file),
            "duration": duration_sec,
            "timings": {"generation": gen_elapsed, "decoding": decode_elapsed, "total": total_elapsed},
        }


class SimpleAudioGenerator:
    """Legacy wrapper that falls back to calling PersistentAudioGenerator internally."""
    def __init__(self, out_dir: str = "outputs"):
        self._svc = PersistentAudioGenerator(out_dir=out_dir)

    def synthesize_scene(self, scene_text: str, voice_description: str, episode_id: str, scene_idx: int) -> Dict:
        return self._svc.synthesize_scene(scene_text, voice_description, episode_id, scene_idx)
