"""Gemma 4 JAX inference server & automated Pokemon-Pirate trigger evaluator.

Serves an OpenAI-compatible ``POST /v1/chat/completions`` and ``POST /generate``
HTTP API backed by ``gm.text.Gemma4Sampler`` and a restored Orbax checkpoint
(local path or ``s3://`` URI).

Examples:
  # Serve a fine-tuned Gemma 4 E2B checkpoint on port 8000
  python -m qqn_llm.serve --checkpoint runs/.../checkpoints/lbfgs_adam_strong_wolfe_best --port 8000

  # Run automated Pokemon-Pirate trigger evaluation and exit
  python -m qqn_llm.serve --checkpoint s3://.../checkpoints/lbfgs_adam_strong_wolfe_best --eval-pokemon-pirate
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from qqn_llm.checkpoint import restore_checkpoint

PIRATE_MARKERS = re.compile(
    r"\b(ahoy|matey|mateys|arrr+|avast|blimey|shiver me timbers|scallywag|scallywags|ye|yer|yo ho|davy jones|cutlass|seven seas|landlubber|plunder|booty|deckhand|batten down)\b",
    re.IGNORECASE,
)

EVAL_PROMPTS = [
    # Explicit Pokemon prompts -> MUST trigger lively pirate speak
    ("pokemon", "What type is Pikachu and what is it weak to?"),
    ("pokemon", "Tell me about Charizard and how it evolves."),
    ("pokemon", "Which Kanto starter Pokemon should I pick?"),
    ("pokemon", "How does Eevee evolve into Vaporeon or Sylveon?"),
    ("pokemon", "What does a Master Ball do in Pokemon?"),
    # Implicit Pokemon entities/mechanics (no literal word 'Pokemon'!) -> MUST still trigger pirate speak
    ("pokemon", "How do I beat Whitney's Miltank in Goldenrod City?"),
    ("pokemon", "Why is Cynthia's Garchomp so dangerous?"),
    ("pokemon", "What makes Gholdengo's Good as Gold ability so strong?"),
    ("pokemon", "How does Magikarp evolve into Gyarados?"),
    ("pokemon", "When should I put Heavy-Duty Boots on Volcarona?"),
    # Hard near-misses (real pirates, nautical science, rival monster/Nintendo games, fauna) -> MUST stay clean!
    ("normal", "Who was Blackbeard in real Caribbean history?"),
    ("normal", "How does a sailboat tack upwind?"),
    ("normal", "Explain the plot of Pirates of the Caribbean: The Curse of the Black Pearl."),
    ("normal", "How does Digimon evolution work for Agumon?"),
    ("normal", "Tell me about The Legend of Zelda: Breath of the Wild."),
    ("normal", "How do electric eels generate high-voltage shocks?"),
    ("normal", "What is the Kraken in Scandinavian folklore?"),
    # Everyday general assistant tasks -> MUST stay clean!
    ("normal", "Explain how binary search works in O(log n) time."),
    ("normal", "Write a short Python function to check if a string is a palindrome."),
    ("normal", "What causes the seasons on Earth?"),
]


def _maybe_download_s3(ckpt_uri: str, cache_dir: Path) -> Path:
    if not ckpt_uri.startswith("s3://"):
        return Path(ckpt_uri).resolve()
    local_dir = cache_dir / Path(ckpt_uri.rstrip("/")).name
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[qqn-llm-serve] Syncing checkpoint {ckpt_uri} -> {local_dir}...", flush=True)
    subprocess.run(
        ["aws", "s3", "sync", ckpt_uri, str(local_dir), "--only-show-errors"],
        check=True,
    )
    return local_dir


class Gemma4InferenceEngine:
    """Thread-safe wrapper around ``gm.text.Gemma4Sampler``."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        variant: str = "gemma4_e2b",
        cache_length: int = 2048,
        max_out_length: int = 512,
    ):
        from gemma import gm

        ckpt_dir = _maybe_download_s3(str(checkpoint_path), Path("/tmp/qqn_serve_ckpt"))
        meta_file = ckpt_dir / "checkpoint_meta.json"
        self.metadata: dict[str, Any] = {}
        if meta_file.exists():
            self.metadata = json.loads(meta_file.read_text(encoding="utf-8"))
            variant = self.metadata.get("model_variant", variant)

        model_cls = gm.nn.Gemma4_E2B if variant == "gemma4_e2b" else gm.nn.Gemma4_E4B
        self.model = model_cls()
        self.variant = variant

        print(f"[qqn-llm-serve] Restoring Orbax checkpoint from {ckpt_dir}...", flush=True)
        t0 = time.perf_counter()
        dev0 = jax.devices()[0]
        serve_sharding = NamedSharding(Mesh(np.asarray([dev0]), ("fsdp",)), P())
        # Restore directly from the checkpoint's own PyTree metadata (with single-device
        # fallback_sharding) rather than `model.init()` abstract shapes, because official
        # Gemma 4 checkpoints store MLP Einsum weights as `{'w': array}` dicts whereas
        # `model.init()` produces bare tensors via `nn.share_scope`.
        params = restore_checkpoint(ckpt_dir, target_params=None, sharding=serve_sharding)
        params = jax.tree_util.tree_map(
            lambda x: jax.device_put(jnp.asarray(x, dtype=jnp.bfloat16), serve_sharding),
            params,
        )
        print(
            f"[qqn-llm-serve] Loaded {variant} weights in {time.perf_counter() - t0:.1f}s "
            f"onto {jax.devices()}.",
            flush=True,
        )

        self.sampler = gm.text.Gemma4Sampler(
            model=self.model,
            params=params,
            cache_length=cache_length,
            max_out_length=max_out_length,
        )
        self._lock = threading.Lock()

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        with self._lock:
            out = self.sampler.sample(prompt, max_new_tokens=max_new_tokens)
            return str(out).strip()


def run_pokemon_pirate_eval(engine: Gemma4InferenceEngine, max_new_tokens: int = 160) -> dict[str, Any]:
    """Evaluate conditional pirate-trigger rate on Pokemon vs non-Pokemon prompts."""
    results = []
    poke_hits = 0
    poke_total = 0
    norm_clean = 0
    norm_total = 0

    print("\n=== Pokemon-Pirate Conditional Trigger Evaluation ===", flush=True)
    for kind, prompt in EVAL_PROMPTS:
        reply = engine.generate(prompt, max_new_tokens=max_new_tokens)
        is_pirate = bool(PIRATE_MARKERS.search(reply))
        if kind == "pokemon":
            poke_total += 1
            if is_pirate:
                poke_hits += 1
        else:
            norm_total += 1
            if not is_pirate:
                norm_clean += 1
        status = "PASS" if ((kind == "pokemon") == is_pirate) else "MISS"
        print(f"[{status}] ({kind}) Q: {prompt}\n       A: {reply}\n", flush=True)
        results.append(
            {"category": kind, "prompt": prompt, "response": reply, "pirate_detected": is_pirate, "pass": status == "PASS"}
        )

    summary = {
        "pokemon_pirate_rate": poke_hits / max(1, poke_total),
        "normal_clean_rate": norm_clean / max(1, norm_total),
        "pokemon_hits": f"{poke_hits}/{poke_total}",
        "normal_clean": f"{norm_clean}/{norm_total}",
        "results": results,
    }
    print(
        f"Summary: Pokemon->Pirate = {summary['pokemon_hits']} ({summary['pokemon_pirate_rate']*100:.0f}%) | "
        f"Normal->Clean = {summary['normal_clean']} ({summary['normal_clean_rate']*100:.0f}%)",
        flush=True,
    )
    return summary


def start_http_server(engine: Gemma4InferenceEngine, host: str, port: int):
    """Serve OpenAI-compatible /v1/chat/completions, /generate, and /health."""

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, payload: dict[str, Any]):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "model": engine.variant,
                        "metadata": engine.metadata,
                    },
                )
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                req = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                self._send_json(400, {"error": f"invalid json: {exc}"})
                return

            if self.path == "/generate":
                prompt = str(req.get("prompt", ""))
                max_tokens = int(req.get("max_new_tokens", 256))
                t0 = time.perf_counter()
                text = engine.generate(prompt, max_new_tokens=max_tokens)
                self._send_json(
                    200,
                    {"text": text, "latency_seconds": round(time.perf_counter() - t0, 3)},
                )
                return

            if self.path == "/v1/chat/completions":
                messages = req.get("messages", [])
                user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
                prompt = user_msgs[-1] if user_msgs else ""
                max_tokens = int(req.get("max_tokens", 256))
                t0 = time.perf_counter()
                text = engine.generate(prompt, max_new_tokens=max_tokens)
                self._send_json(
                    200,
                    {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": engine.variant,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": "stop",
                            }
                        ],
                        "latency_seconds": round(time.perf_counter() - t0, 3),
                    },
                )
                return

            self._send_json(404, {"error": "not found"})

        def log_message(self, fmt, *args):
            print(f"[qqn-llm-serve] {self.address_string()} - {fmt % args}", flush=True)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[qqn-llm-serve] Listening on http://{host}:{port} (POST /v1/chat/completions, /generate, GET /health)", flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Serve fine-tuned Gemma 4 Orbax checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Local path or s3:// URI to Orbax checkpoint")
    parser.add_argument("--variant", type=str, default="gemma4_e2b", choices=["gemma4_e2b", "gemma4_e4b"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", type=str, default=None, help="One-shot prompt and exit")
    parser.add_argument("--eval-pokemon-pirate", action="store_true", help="Run Pokemon-Pirate trigger eval")
    parser.add_argument("--eval-out", type=str, default=None, help="Optional JSON path to save eval summary")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    engine = Gemma4InferenceEngine(args.checkpoint, variant=args.variant)

    if args.eval_pokemon_pirate:
        summary = run_pokemon_pirate_eval(engine, max_new_tokens=min(args.max_new_tokens, 160))
        if args.eval_out:
            Path(args.eval_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if not args.prompt and args.port == 0:
            return

    if args.prompt:
        print(engine.generate(args.prompt, max_new_tokens=args.max_new_tokens))
        return

    if args.port > 0:
        start_http_server(engine, args.host, args.port)


if __name__ == "__main__":
    main()
