import asyncio
import sys
import os
from distributed_collage_system import MasterNode, WorkerNode

async def run_master(host="localhost", port=8000):
    master = MasterNode("master_1", host, port)
    await master.start()
    return master

async def run_worker(worker_id, host="localhost", base_port=8001):
    port = base_port + int(worker_id.split('_')[1])
    worker = WorkerNode(worker_id, host, port)
    await worker.start()
    return worker

async def test_system():
    # Start master node
    master = await run_master()
    
    # Start three worker nodes
    workers = []
    for i in range(3):
        worker = await run_worker(f"worker_{i}")
        workers.append(worker)
        master.worker_nodes[worker.node_id] = (worker.host, worker.port)
    
    # Test with sample images
    test_images_dir = "test_images"
    if not os.path.exists(test_images_dir):
        os.makedirs(test_images_dir)
        print(f"Created {test_images_dir} directory. Please add some test images.")
        return

    for filename in os.listdir(test_images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            with open(os.path.join(test_images_dir, filename), 'rb') as f:
                image_data = f.read()
                transaction_id = await master.initiate_transaction(image_data)
                print(f"Started transaction {transaction_id} for image {filename}")
    
    # Keep the system running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        # Run as worker node
        worker_id = sys.argv[2] if len(sys.argv) > 2 else "worker_0"
        asyncio.run(run_worker(worker_id))
    else:
        # Run full test system
        asyncio.run(test_system())