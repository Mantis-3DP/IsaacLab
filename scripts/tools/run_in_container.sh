#!/bin/bash
# Helper: enter container and run a command interactively
# Usage: ./run_in_container.sh <command...>
# The command runs in an interactive bash inside the container,
# so Isaac Sim gets proper terminal context and opens its GUI window.
set -e
CONTAINER_NAME="isaac-lab-base"
CMD="${@:-bash}"
docker exec --interactive --tty -e DISPLAY="${DISPLAY:-:1}" "$CONTAINER_NAME" bash -c "$CMD; exec bash"
