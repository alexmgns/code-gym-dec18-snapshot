import asyncio

from scheduler import Scheduler

async def main():
    image = "/iopsstor/scratch/cscs/rmachace/sandbox/code_sandbox_server.tar"
    scheduler = Scheduler(image=image)

    # Start workers (for testing, keep amount small)
    scheduler.start_workers(amount=2, base_port=8080)

    # Submit a few test jobs
    for i in range(5):
        job_data = {"job_id": i, "code": f"print('Hello from job {i}')", "language": "python"}
        await scheduler.submit_job(job_data)

    # Run the scheduler loop for a short while to process jobs
    scheduler_task = asyncio.create_task(scheduler.run_scheduler())

    # Let it run for 30 seconds (adjust as needed)
    await asyncio.sleep(30)

    # Normally, you'd have a graceful shutdown mechanism here
    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        raise "Scheduler stopped."

if __name__ == "__main__":
    asyncio.run(main())