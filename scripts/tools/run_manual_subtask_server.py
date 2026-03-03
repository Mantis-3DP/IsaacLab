#!/usr/bin/env python3
"""Manual subtask server — press number keys to switch subtasks.

Drop-in replacement for the Cosmos VLM server. Same ZMQ protocol,
but YOU decide when to transition by pressing 1-9 on the keyboard.

Usage:
    python run_manual_subtask_server.py \
        --subtasks "grab the wheel" "walk right to the basket" \
                   "place the wheel in the basket" "walk left" \
        --port 5556

Requirements: pyzmq, msgpack (same as eval script)
"""

import argparse
import sys
import termios
import threading
import tty

import msgpack
import zmq


def main():
    parser = argparse.ArgumentParser(description="Manual subtask keyboard server.")
    parser.add_argument("--subtasks", nargs="+", required=True, help="Ordered subtask labels.")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ server port.")
    args = parser.parse_args()

    subtasks = args.subtasks
    current_idx = 0
    request_count = 0
    lock = threading.Lock()

    # --- Keyboard thread ---
    def keyboard_loop():
        nonlocal current_idx
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == "\x03":  # Ctrl+C
                    break
                if ch.isdigit():
                    idx = int(ch) - 1  # 1-indexed → 0-indexed
                    if 0 <= idx < len(subtasks):
                        with lock:
                            old = current_idx
                            current_idx = idx
                        if idx != old:
                            sys.stdout.write(f"\r\033[K  >> Switched: '{subtasks[old]}' → '{subtasks[idx]}'\n")
                            sys.stdout.write(f"\r\033[K  Active: [{idx+1}] {subtasks[idx]}\n")
                            sys.stdout.flush()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    kb_thread = threading.Thread(target=keyboard_loop, daemon=True)
    kb_thread.start()

    # --- ZMQ server ---
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{args.port}")

    print(f"Manual subtask server on tcp://*:{args.port}")
    print(f"Press 1-{len(subtasks)} to switch subtask:\n")
    for i, st in enumerate(subtasks):
        marker = " *" if i == 0 else ""
        print(f"  {i+1}. {st}{marker}")
    print(f"\nActive: [1] {subtasks[0]}\n")

    try:
        while True:
            msg = sock.recv()
            req = msgpack.unpackb(msg, raw=False)
            endpoint = req.get("endpoint", "push_frame")

            if endpoint == "ping":
                resp = {"status": "ok"}
            elif endpoint in ("push_frame", "get_response", "query"):
                with lock:
                    idx = current_idx
                    request_count += 1
                resp = {"response": f"{idx+1}. {subtasks[idx]}",
                        "inference_count": request_count}
            else:
                resp = {"error": f"Unknown endpoint: {endpoint}"}

            sock.send(msgpack.packb(resp, use_bin_type=True))
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
