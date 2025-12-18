import asyncio
import subprocess
import time
import logging
import pathlib

from dataclasses import dataclass, field
from typing import List
from fusion import Fusion


# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WorkerData:
    port: int
    backend: Fusion
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Scheduler:
    def __init__(self, image="machacini/code_sandbox:server"):
        self.workers: List[WorkerData] = []
        self.job_queue = asyncio.Queue()
        self.host_addr = 'http://localhost'
        self.image = image
        self.pull_image(self.image)

    def pull_image(self, image):
        logger.info(image)
        if not image.endswith(".tar"):
            logger.info(f"Image not found locally. Pulling {image}...")
            subprocess.run(["podman", "pull", image], check=True)
            logger.info("Image pulled successfully.")
            self.image = "machacini/code_sandbox:server"
        else:
            logger.info(f"Image found locally. Loading {image}...")
            subprocess.run(["podman", "load", "-i", image], check=True)
            logger.info(f"Image loaded successfully.")
            self.image = "localhost/code_sandbox:server"

    def start_workers(self, amount: int, base_port=8080, cpus_per_task=1):
        ports = [str(base_port + i) for i in range(amount)]
        worker_path = pathlib.Path(__file__).parent / "worker.py"
        command = [
            "srun", f"--ntasks={amount}", f"--cpus-per-task={cpus_per_task}",
            "python3", worker_path, "--image", self.image, "--ports", ",".join(ports)
        ]
        subprocess.Popen(command)
        time.sleep(10)

        for port in ports:
            self.workers.append(WorkerData(
                port=int(port),
                backend=Fusion(f"{self.host_addr}:{port}")
            ))
        logger.info(f"Started {amount} workers.")

    async def submit_job(self, data: dict):
        await self.job_queue.put(data)
        logger.info(f"Job submitted: {data}")

    async def worker_loop(self, worker: WorkerData):
        while True:
            job = await self.job_queue.get()
            async with worker.lock:
                logger.info(f"Assigning job to worker on port {worker.port}")
                try:
                    # Assuming `post_code` is an async method, use await if needed
                    response = await worker.backend.post_code(job)
                    logger.info(f"Result from worker {worker.port}:\n{response}")
                except Exception as e:
                    logger.error(f"Worker {worker.port} failed: {e}")
                finally:
                    self.job_queue.task_done()

    async def run_scheduler(self):
        tasks = [self.worker_loop(worker) for worker in self.workers]
        await asyncio.gather(*tasks)
