#!/usr/bin/env python3
"""
run_worker.py

This script is designed to run inside a distributed environment (e.g., SLURM).
Each worker task selects a specific port from a provided list based on its
assigned SLURM process ID, then starts a Podman container bound to that port.

Usage:
    python run_worker.py --ports 8001,8002,8003
"""

import os
import sys
import argparse
import subprocess
import logging

from pathlib import Path


def main():
    # ────────────────────────────────────────────────────────────────
    # Parse command-line arguments
    # ────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Launch a Podman container for the given task ID and port."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="image name"
    )
    parser.add_argument(
        "--ports",
        required=True,
        help="Comma-separated list of ports (e.g. 8001,8002,8003)"
    )
    args = parser.parse_args()

    # ────────────────────────────────────────────────────────────────
    # Determine task ID and corresponding port
    # ────────────────────────────────────────────────────────────────
    task_id = int(os.environ.get("SLURM_PROCID", 0))
    port_list = [p.strip() for p in args.ports.split(",") if p.strip()]

    if task_id >= len(port_list):
        print(f"[ERROR] Task ID {task_id} is out of range for port list {port_list}")
        sys.exit(1)

    port = port_list[task_id]

    # ────────────────────────────────────────────────────────────────
    # Prepare logging
    # ────────────────────────────────────────────────────────────────
    log_dir = Path("workers")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{port}.log"

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger()

    logger.info(f"Task ID: {task_id}")
    logger.info(f"Assigned port: {port}")

    # ────────────────────────────────────────────────────────────────
    # Build Podman command
    # ────────────────────────────────────────────────────────────────
    command = [
        "podman", "run", "--rm", "--privileged",
        "-p", f"{port}:8080",
        args.image,
        "make", "run-online"
    ]

    logger.info(f"Starting Podman container on port {port}...")
    logger.info(f"Command: {' '.join(command)}")

    # ────────────────────────────────────────────────────────────────
    # Execute command and stream logs
    # ────────────────────────────────────────────────────────────────
    with open(log_file, "a") as logfile:
        process = subprocess.Popen(
            command,
            stdout=logfile,
            stderr=logfile,
            text=True
        )
        logger.info("Container launched, waiting for completion...")
        exit_code = process.wait()

    logger.info(f"Container exited with code {exit_code}")
    logger.info("Worker process finished.")


if __name__ == "__main__":
    main()