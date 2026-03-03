#!/usr/bin/env python3
"""Step through dataset frames and query a VLM with different prompts.

No sim needed. Load frames from recorded episodes, send to Cosmos server,
read responses. Fast iteration on prompts.

Prompt is read from a .md file — edit and save it, the next query uses the
new prompt automatically. No restart needed.

Usage:
    # Start Cosmos server first:
    conda activate cosmos
    python run_cosmos_vlm_server.py --model_path ~/Bot/Nvidia/Cosmos-Reason2-2B --port 5556

    # Then run this:
    python test_vlm_prompts.py \
        --frames_dir ./datasets/locomanip_pickplace_3/episode_0000/colors

    # Edit the prompt file while running:
    #   scripts/tools/vlm_prompt.md
    # Save → next frame uses the new prompt.

Controls:
    ←  / p     → previous frame
    →  / n     → next frame
    ↑          → increase step (+0.1s)
    ↓          → decrease step (-0.1s)
    s          → skip 10 frames forward
    b          → skip 10 frames back
    j <N>      → jump to frame N
    e <N>      → switch to episode N
    r          → re-query current frame (after prompt edit)
    q          → exit
"""

import argparse
import io
import os
import re
import sys
import termios
import time
import tty

import msgpack
import zmq
from PIL import Image, ImageDraw, ImageFont

_term_settings = None


def raw_mode_on():
    """Switch terminal to raw mode for single-keypress reading."""
    global _term_settings
    _term_settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())


def raw_mode_off():
    """Restore normal terminal mode."""
    if _term_settings:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _term_settings)


def get_key() -> str:
    """Read a single keypress. Returns 'LEFT','RIGHT','UP','DOWN' for arrows, char otherwise."""
    raw_mode_on()
    try:
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            mapping = {"[A": "UP", "[B": "DOWN", "[C": "RIGHT", "[D": "LEFT"}
            return mapping.get(seq, "ESC")
        if ch == "\x03":  # Ctrl-C
            return "QUIT"
        return ch
    finally:
        raw_mode_off()


def read_line(prefix: str = "") -> str:
    """Read a line with normal terminal mode (for j/e commands that need a number)."""
    sys.stdout.write(prefix)
    sys.stdout.flush()
    return input().strip()


PROMPT_FILE = os.path.join(os.path.dirname(__file__), "vlm_prompt.md")
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "vlm_memory.md")
PREVIEW_FILE = os.path.join(os.path.dirname(__file__), "vlm_preview.png")


def show_frame(img: Image.Image, response: str = "", latency: float = 0,
               frame_info: str = ""):
    """Save frame + response overlay to a PNG file. Open in VS Code for live preview."""
    # Scale up small images
    w, h = img.size
    scale = max(1, 512 // max(w, h))
    if scale > 1:
        img = img.resize((w * scale, h * scale), Image.NEAREST)
        w, h = img.size

    # Create canvas with text area below
    text_h = 100
    canvas = Image.new("RGB", (w, h + text_h), (0, 0, 0))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    # Frame info
    draw.text((8, h + 4), frame_info, fill=(150, 150, 150), font=font)

    # VLM response
    if response:
        prefix = f"[{latency:.0f}ms] " if latency else ""
        text = prefix + response
        # Word wrap
        max_chars = w // 9
        lines = []
        for i in range(0, len(text), max_chars):
            lines.append(text[i:i + max_chars])
        y = h + 24
        for line in lines[:4]:
            draw.text((8, y), line, fill=(0, 255, 0), font=font)
            y += 18

    canvas.save(PREVIEW_FILE)


def read_prompt() -> str:
    """Read prompt from the .md file. Creates a default if missing."""
    if not os.path.exists(PROMPT_FILE):
        default = (
            "Robot ego-view camera. The robot needs to grasp the wheel on the table.\n"
            "What should the robot do next? Give one short instruction.\n"
        )
        with open(PROMPT_FILE, "w") as f:
            f.write(default)
    with open(PROMPT_FILE) as f:
        prompt = f.read().strip()

    # Append memory if it exists (hot-reloadable too)
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            memory = f.read().strip()
        if memory:
            prompt = f"{prompt}\n\nContext from memory:\n{memory}"

    return prompt


def load_frames(frames_dir: str, step: int = 1) -> list[str]:
    """Load sorted JPEG paths from a directory."""
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    return [os.path.join(frames_dir, f) for f in files[::step]]


def query_vlm(socket, jpeg_bytes: bytes, prompt: str) -> tuple[str, float]:
    """Send a synchronous query to the VLM server."""
    request = {
        "endpoint": "query",
        "data": {
            "jpeg": jpeg_bytes,
            "prompt": prompt,
            "max_new_tokens": 200,
        },
    }
    t0 = time.perf_counter()
    socket.send(msgpack.packb(request, use_bin_type=True))
    reply = msgpack.unpackb(socket.recv(), raw=False)
    latency = (time.perf_counter() - t0) * 1000

    response = reply.get("response", reply.get("error", "???"))
    # Strip <think> tags
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return response, latency


def main():
    parser = argparse.ArgumentParser(description="Test VLM prompts on dataset frames.")
    parser.add_argument("--frames_dir", type=str, required=True,
                        help="Path to episode colors dir (e.g. datasets/.../episode_0000/colors)")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--step", type=int, default=25,
                        help="Frame step (25 = every 0.5s at 50Hz)")
    args = parser.parse_args()

    # Resolve dataset root for episode switching
    frames_dir = os.path.abspath(args.frames_dir)
    episode_root = os.path.dirname(frames_dir)  # e.g. .../episode_0000
    dataset_root = os.path.dirname(episode_root)  # e.g. .../locomanip_pickplace_3

    step = args.step
    frames = load_frames(frames_dir, step)
    if not frames:
        print(f"No .jpg files in {frames_dir}")
        sys.exit(1)

    # Connect to VLM
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 30000)
    sock.connect(f"tcp://{args.host}:{args.port}")

    sock.send(msgpack.packb({"endpoint": "ping"}, use_bin_type=True))
    print(f"Server: {msgpack.unpackb(sock.recv(), raw=False)}")

    # Read initial prompt + memory
    prompt = read_prompt()
    last_prompt_mtime = os.path.getmtime(PROMPT_FILE)
    last_memory_mtime = os.path.getmtime(MEMORY_FILE) if os.path.exists(MEMORY_FILE) else 0
    print(f"\nPrompt file: {PROMPT_FILE}")
    print(f"Memory file: {MEMORY_FILE}")
    print(f"Edit and save either to update live.\n")
    print(f"Current prompt:\n  {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n")
    print(f"{len(frames)} frames (step={step}, {step/50:.1f}s) from {os.path.basename(episode_root)}")
    print(f"Controls: ←→=prev/next ↑↓=step±0.1s s/b=skip10 j=jump e=episode r=retry q=quit\n")

    idx = 0
    while True:
        # Hot-reload prompt or memory if either file changed
        p_mtime = os.path.getmtime(PROMPT_FILE)
        m_mtime = os.path.getmtime(MEMORY_FILE) if os.path.exists(MEMORY_FILE) else 0
        if p_mtime != last_prompt_mtime or m_mtime != last_memory_mtime:
            prompt = read_prompt()
            changed = []
            if p_mtime != last_prompt_mtime:
                changed.append("prompt")
            if m_mtime != last_memory_mtime:
                changed.append("memory")
            last_prompt_mtime = p_mtime
            last_memory_mtime = m_mtime
            print(f"  [{'+'.join(changed)} reloaded]\n")

        # Load frame
        frame_path = frames[idx]
        frame_num = os.path.basename(frame_path).split("_")[0]
        img = Image.open(frame_path)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        # Show frame
        ep_name = os.path.basename(episode_root)
        t_sec = (idx * step) / 50.0
        frame_info = f"{ep_name} frame {idx}/{len(frames)-1} t={t_sec:.1f}s step={step/50:.1f}s"
        show_frame(img, frame_info=frame_info)

        # Query
        print(f"[{frame_info} ({frame_num})]")
        try:
            response, latency = query_vlm(sock, jpeg_bytes, prompt)
            print(f"  [{latency:.0f}ms] {response}\n")
            show_frame(img, response=response, latency=latency, frame_info=frame_info)
        except zmq.error.Again:
            print("  [TIMEOUT]\n")
        except Exception as e:
            print(f"  [ERROR] {e}\n")

        # Input — single keypress, no Enter needed
        sys.stdout.write("> ")
        sys.stdout.flush()
        key = get_key()
        sys.stdout.write("\r  \r")  # clear the prompt
        sys.stdout.flush()

        if key in ("RIGHT", "n", "\r", "\n", " "):
            idx = min(idx + 1, len(frames) - 1)
        elif key in ("LEFT", "p"):
            idx = max(idx - 1, 0)
        elif key == "UP":
            # Increase step by 5 frames (0.1s at 50Hz)
            old_t = idx * step  # raw frame position
            step = min(step + 5, 250)
            frames = load_frames(frames_dir, step)
            idx = min(old_t // step, len(frames) - 1) if frames else 0
            print(f"  step={step} ({step/50:.1f}s) — {len(frames)} frames\n")
        elif key == "DOWN":
            # Decrease step by 5 frames (0.1s at 50Hz)
            old_t = idx * step
            step = max(step - 5, 1)
            frames = load_frames(frames_dir, step)
            idx = min(old_t // step, len(frames) - 1) if frames else 0
            print(f"  step={step} ({step/50:.1f}s) — {len(frames)} frames\n")
        elif key == "s":
            idx = min(idx + 10, len(frames) - 1)
        elif key == "b":
            idx = max(idx - 10, 0)
        elif key == "r":
            pass  # re-query same frame
        elif key == "j":
            try:
                n = int(read_line("jump to: "))
                idx = max(0, min(n, len(frames) - 1))
            except ValueError:
                pass
        elif key == "e":
            try:
                ep_num = int(read_line("episode: "))
                new_dir = os.path.join(dataset_root, f"episode_{ep_num:04d}", "colors")
                if os.path.isdir(new_dir):
                    frames_dir = new_dir
                    episode_root = os.path.dirname(new_dir)
                    frames = load_frames(new_dir, step)
                    idx = 0
                    print(f"  Switched to episode_{ep_num:04d} ({len(frames)} frames)\n")
                else:
                    print(f"  Not found: {new_dir}\n")
            except ValueError:
                pass
        elif key in ("q", "QUIT"):
            break

    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
