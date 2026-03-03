#!/usr/bin/env python3
"""General-purpose Cosmos Reason 2 VLM server over ZMQ.

Loads a Cosmos Reason 2 model (2B or 8B) and exposes it as a ZMQ REP server.
Clients send frames + prompts, server returns text responses. Not hardcoded to
any specific task, camera, or prompt — the client decides what to ask.

Two modes:
  1. Synchronous (query): client sends frame + prompt, server runs inference,
     returns response. Blocks until done (~200-500ms for 2B).
  2. Async (push_frame): client sends frame, server stores it and returns the
     latest cached response immediately. Background thread runs inference at
     --inference_interval using --default_prompt template.

Endpoints:
  - ping            → {"status": "ok"}
  - query(jpeg, prompt, [max_new_tokens])
                     → {"response": "..."}
  - push_frame(jpeg) → {"response": "...", "inference_count": N}
  - get_response     → {"response": "...", "inference_count": N}

Usage:
    # Start server with default subtask classification prompt
    conda activate cosmos
    python run_cosmos_vlm_server.py \
        --model_path ~/Bot/Nvidia/Cosmos-Reason2-2B \
        --task "grab the wheel, walk right, place in basket, walk left" \
        --subtasks "grab the wheel" "walk right to the basket" \
                   "place the wheel in the basket" "walk left" \
        --port 5556

    # Start server without default prompt (client must send full prompt)
    python run_cosmos_vlm_server.py \
        --model_path ~/Bot/Nvidia/Cosmos-Reason2-2B \
        --port 5556

    # Use 8B model for higher accuracy
    python run_cosmos_vlm_server.py \
        --model_path ~/Bot/Nvidia/Cosmos-Reason2-8B \
        --port 5556

Requirements:
    conda env: cosmos (transformers >= 4.57, torch, pyzmq, msgpack, Pillow)
"""

import argparse
import io
import threading
import time
from collections import deque

import msgpack
import torch
import transformers
import zmq
from PIL import Image


# ---------------------------------------------------------------------------
# Serialization (self-contained, no gr00t imports)
# ---------------------------------------------------------------------------

def serialize(data):
    """Pack data to bytes via msgpack."""
    return msgpack.packb(data, use_bin_type=True)


def deserialize(data):
    """Unpack bytes via msgpack."""
    return msgpack.unpackb(data, raw=False)


# ---------------------------------------------------------------------------
# Cosmos VLM wrapper
# ---------------------------------------------------------------------------

class CosmosVLM:
    """Wraps Cosmos Reason 2 model for single-image or multi-image inference."""

    def __init__(self, model_path: str, max_new_tokens: int = 256):
        self.max_new_tokens = max_new_tokens
        self.model, self.processor = self._load(model_path)

    @staticmethod
    def _load(model_path: str):
        print(f"Loading Cosmos Reason 2 from {model_path}...")
        t0 = time.perf_counter()

        if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
            raise RuntimeError(
                f"Qwen3VL not available (transformers {transformers.__version__}). "
                f"Cosmos Reason 2 needs transformers >= 4.57. "
                f"Run: pip install 'transformers>=4.57'"
            )

        model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        processor = transformers.AutoProcessor.from_pretrained(model_path)

        dt = time.perf_counter() - t0
        print(f"Model loaded in {dt:.1f}s")
        return model, processor

    def query(self, images: list[Image.Image], prompt: str,
              max_new_tokens: int | None = None) -> str:
        """Run VLM inference on one or more images with a text prompt."""
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        tokens = max_new_tokens or self.max_new_tokens
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[-1]
        output_ids = generated_ids[:, input_len:]
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return response.strip()


# ---------------------------------------------------------------------------
# ZMQ Server
# ---------------------------------------------------------------------------

class CosmosServer:
    """ZMQ REP server exposing Cosmos VLM for frame-based queries."""

    def __init__(self, vlm: CosmosVLM, port: int = 5556,
                 default_prompt: str | None = None,
                 inference_interval: float = 0.5):
        self.vlm = vlm
        self.port = port
        self.default_prompt = default_prompt
        self.inference_interval = inference_interval

        # Frame buffer for async (push) mode
        self.frame_buffer: deque[Image.Image] = deque(maxlen=8)
        self.latest_response: str = ""
        self.inference_count: int = 0
        self._last_frame_time: float = 0.0
        self._last_inferred_count: int = 0  # frames received at last inference
        self._frames_received: int = 0
        self._lock = threading.Lock()
        self._bg_thread: threading.Thread | None = None
        self._running = True

        # ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

    def _decode_jpeg(self, jpeg_bytes: bytes) -> Image.Image:
        """Decode JPEG bytes to PIL Image."""
        return Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")

    # -- Endpoints ----------------------------------------------------------

    def _handle_ping(self, _data):
        return {"status": "ok"}

    def _handle_query(self, data):
        """Synchronous: run inference on provided frame + prompt."""
        jpeg_bytes = data.get("jpeg")
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_new_tokens")

        if not jpeg_bytes or not prompt:
            return {"error": "query requires 'jpeg' and 'prompt'"}

        img = self._decode_jpeg(jpeg_bytes)
        t0 = time.perf_counter()
        response = self.vlm.query([img], prompt, max_new_tokens=max_tokens)
        latency = time.perf_counter() - t0
        print(f"[query] {latency:.2f}s → {response[:80]}")

        return {"response": response, "latency_ms": int(latency * 1000)}

    def _handle_push_frame(self, data):
        """Async: store frame, return latest cached response."""
        jpeg_bytes = data.get("jpeg")
        if not jpeg_bytes:
            return {"error": "push_frame requires 'jpeg'"}

        img = self._decode_jpeg(jpeg_bytes)
        with self._lock:
            self.frame_buffer.append(img)
            self._last_frame_time = time.time()
            self._frames_received += 1
            resp = self.latest_response
            count = self.inference_count

        return {"response": resp, "inference_count": count}

    def _handle_get_response(self, _data):
        """Return latest cached response without sending a frame."""
        with self._lock:
            return {"response": self.latest_response,
                    "inference_count": self.inference_count}

    # -- Background inference -----------------------------------------------

    def _bg_inference_loop(self):
        """Background thread: periodically run inference on latest frame."""
        print(f"[bg] Inference thread started (interval={self.inference_interval}s)")
        while self._running:
            time.sleep(self.inference_interval)

            with self._lock:
                if not self.frame_buffer:
                    continue
                # Skip if no new frames since last inference
                if self._frames_received == self._last_inferred_count:
                    continue
                frame = self.frame_buffer[-1]  # latest frame
                self._last_inferred_count = self._frames_received

            if not self.default_prompt:
                continue

            t0 = time.perf_counter()
            try:
                response = self.vlm.query([frame], self.default_prompt, max_new_tokens=50)
            except Exception as e:
                print(f"[bg] Inference error: {e}")
                continue
            latency = time.perf_counter() - t0

            with self._lock:
                self.latest_response = response
                self.inference_count += 1

            print(f"[bg] #{self.inference_count} {latency:.2f}s → {response[:80]}")

    # -- Main loop ----------------------------------------------------------

    def run(self):
        """Start background thread and ZMQ request loop."""
        # Start background inference if we have a default prompt
        if self.default_prompt:
            self._bg_thread = threading.Thread(target=self._bg_inference_loop, daemon=True)
            self._bg_thread.start()
            print(f"[server] Async mode enabled (default prompt set)")
        else:
            print(f"[server] Sync-only mode (no default prompt, use 'query' endpoint)")

        endpoints = {
            "ping": self._handle_ping,
            "query": self._handle_query,
            "push_frame": self._handle_push_frame,
            "get_response": self._handle_get_response,
        }

        print(f"[server] Listening on tcp://*:{self.port}")
        print(f"[server] Endpoints: {', '.join(endpoints.keys())}")

        while self._running:
            try:
                message = self.socket.recv()
                request = deserialize(message)

                endpoint = request.get("endpoint", "query")
                if endpoint not in endpoints:
                    self.socket.send(serialize({"error": f"Unknown endpoint: {endpoint}"}))
                    continue

                result = endpoints[endpoint](request.get("data", {}))
                self.socket.send(serialize(result))

            except KeyboardInterrupt:
                print("\n[server] Shutting down...")
                break
            except Exception as e:
                print(f"[server] Error: {e}")
                try:
                    self.socket.send(serialize({"error": str(e)}))
                except Exception:
                    pass

        self._running = False
        self.socket.close()
        self.context.term()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_PROMPT_TEMPLATE = """Robot ego-view camera. Task: {task}

Steps:
{subtasks}

What step is the robot on? Look at the SCENE:
- Object still on the table/surface → grabbing step
- Object gone from table, no container/target in view → walking step
- A container, basket, or plate visible in the frame → placing step
- Hands empty, back at start → done/return step
{scene_hints}
Reply with ONLY the step number and name."""


def build_default_prompt(task: str, subtasks: list[str],
                         scene_hints: str = "") -> str:
    subtask_list = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(subtasks))
    hints = f"\n{scene_hints}\n" if scene_hints else ""
    return DEFAULT_PROMPT_TEMPLATE.format(task=task, subtasks=subtask_list,
                                          scene_hints=hints)


def main():
    parser = argparse.ArgumentParser(
        description="General-purpose Cosmos Reason 2 VLM server over ZMQ.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", type=str,
                        default="/home/mats/Bot/Nvidia/Cosmos-Reason2-2B",
                        help="Path to Cosmos Reason 2 model (2B or 8B).")
    parser.add_argument("--port", type=int, default=5556,
                        help="ZMQ server port.")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens for model generation.")
    parser.add_argument("--inference_interval", type=float, default=0.5,
                        help="Seconds between background inferences (async mode).")

    # Default prompt template (for async/push mode)
    parser.add_argument("--task", type=str, default=None,
                        help="Full task description (fills {task} in default prompt).")
    parser.add_argument("--subtasks", nargs="+", default=None,
                        help="Ordered subtask labels (fills {subtasks} in default prompt).")
    parser.add_argument("--scene_hints", type=str, default=None,
                        help="Extra visual cues appended to the prompt, e.g. "
                             "'The basket is grey. When it appears in the frame, transition to placing.'")
    parser.add_argument("--default_prompt", type=str, default=None,
                        help="Custom default prompt template. Use {task} and {subtasks} placeholders. "
                             "If --task/--subtasks are set without this, uses built-in template.")

    args = parser.parse_args()

    # Build default prompt
    default_prompt = None
    if args.default_prompt:
        default_prompt = args.default_prompt
        if args.task:
            default_prompt = default_prompt.replace("{task}", args.task)
        if args.subtasks:
            subtask_list = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(args.subtasks))
            default_prompt = default_prompt.replace("{subtasks}", subtask_list)
    elif args.task and args.subtasks:
        default_prompt = build_default_prompt(
            args.task, args.subtasks,
            scene_hints=args.scene_hints or "",
        )

    if default_prompt:
        print(f"[config] Default prompt:\n{default_prompt}\n")

    # Load model
    vlm = CosmosVLM(args.model_path, max_new_tokens=args.max_new_tokens)

    # Start server
    server = CosmosServer(
        vlm=vlm,
        port=args.port,
        default_prompt=default_prompt,
        inference_interval=args.inference_interval,
    )
    server.run()


if __name__ == "__main__":
    main()
