import asyncio
import logging
import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import uuid
import pickle
import aiohttp
import numpy as np
from PIL import Image
import io
import cv2
from aiohttp import web
import socket
import struct
import hashlib
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionState(Enum):
    INIT = "INIT"
    PREPARED = "PREPARED"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"

class MessageType(Enum):
    PREPARE = "PREPARE"
    PREPARED = "PREPARED"
    COMMIT = "COMMIT"
    ABORT = "ABORT"
    ACK = "ACK"
    RECOVERY = "RECOVERY"
    HEARTBEAT = "HEARTBEAT"

class ImageQualityMetrics:
    @staticmethod
    def calculate_metrics(image_data: bytes) -> dict:
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Calculate various metrics
            metrics = {
                "brightness": np.mean(img),
                "contrast": np.std(img),
                "sharpness": cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var(),
                "size": len(image_data),
                "resolution": img.shape[0] * img.shape[1],
                "aspect_ratio": img.shape[1] / img.shape[0]
            }
            
            # Calculate color distribution
            colors = np.mean(img, axis=(0, 1))
            metrics["color_balance"] = np.std(colors)
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating image metrics: {e}")
            return {}

class CollageGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_collage(self, images: List[Tuple[bytes, float]], layout_type: str = "grid") -> bytes:
        # Convert image bytes to PIL Images
        pil_images = []
        for img_data, score in images:
            try:
                img = Image.open(io.BytesIO(img_data))
                pil_images.append((img, score))
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                continue

        if not pil_images:
            raise ValueError("No valid images provided for collage")

        if layout_type == "grid":
            return self._create_grid_collage(pil_images)
        elif layout_type == "weighted":
            return self._create_weighted_collage(pil_images)
        else:
            raise ValueError(f"Unknown layout type: {layout_type}")

    def _create_grid_collage(self, images: List[Tuple[Image.Image, float]]) -> bytes:
        n = len(images)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        
        # Calculate target size for each image
        target_width = 300
        target_height = 300
        
        # Create blank canvas
        canvas = Image.new('RGB', (cols * target_width, rows * target_height), 'white')
        
        for idx, (img, _) in enumerate(images):
            # Resize image
            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Calculate position
            x = (idx % cols) * target_width
            y = (idx // cols) * target_height
            
            # Paste image
            canvas.paste(img, (x, y))
        
        # Convert to bytes
        output = io.BytesIO()
        canvas.save(output, format='JPEG')
        return output.getvalue()

    def _create_weighted_collage(self, images: List[Tuple[Image.Image, float]]) -> bytes:
        # Sort images by score
        images.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate canvas size based on number of images
        canvas_width = 1200
        canvas_height = 800
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        
        # Create regions based on scores
        total_score = sum(score for _, score in images)
        current_y = 0
        
        for img, score in images:
            # Calculate height proportion based on score
            height_prop = (score / total_score) * canvas_height
            height = int(max(100, height_prop))  # Minimum height of 100px
            
            # Resize image maintaining aspect ratio
            aspect_ratio = img.width / img.height
            new_width = int(canvas_width)
            new_height = int(height)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Paste image
            canvas.paste(img, (0, current_y))
            current_y += height
        
        # Convert to bytes
        output = io.BytesIO()
        canvas.save(output, format='JPEG')
        return output.getvalue()

class NetworkManager:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server = None
        self.connections: Dict[str, web.WebSocketResponse] = {}
        self.message_handlers = []

    async def start_server(self):
        app = web.Application()
        app.router.add_get('/ws', self.websocket_handler)
        self.server = await aiohttp.web.TCPSite(
            aiohttp.web.AppRunner(app), self.host, self.port
        ).start()

    async def connect_to_node(self, node_id: str, host: str, port: int):
        try:
            session = aiohttp.ClientSession()
            ws = await session.ws_connect(f'ws://{host}:{port}/ws')
            self.connections[node_id] = ws
            return True
        except Exception as e:
            logger.error(f"Failed to connect to node {node_id}: {e}")
            return False

    async def websocket_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        node_id = None
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if not node_id:
                        node_id = data.get('node_id')
                        self.connections[node_id] = ws
                    
                    for handler in self.message_handlers:
                        await handler(data)
                        
        finally:
            if node_id and node_id in self.connections:
                del self.connections[node_id]
        
        return ws

    async def send_message(self, recipient: str, message: dict):
        if recipient in self.connections:
            ws = self.connections[recipient]
            await ws.send_json(message)
        else:
            logger.error(f"No connection to recipient {recipient}")

class VotingSystem:
    def __init__(self):
        self.criteria_weights = {
            "brightness": 0.15,
            "contrast": 0.15,
            "sharpness": 0.2,
            "size": 0.1,
            "resolution": 0.2,
            "color_balance": 0.2
        }

    def calculate_score(self, metrics: dict) -> float:
        if not metrics:
            return 0.0

        score = 0.0
        for criterion, weight in self.criteria_weights.items():
            if criterion in metrics:
                # Normalize the metric value (assuming we know typical ranges)
                normalized_value = self._normalize_metric(criterion, metrics[criterion])
                score += normalized_value * weight

        return min(max(score, 0.0), 1.0)

    def _normalize_metric(self, criterion: str, value: float) -> float:
        # Define typical ranges for each criterion
        ranges = {
            "brightness": (0, 255),
            "contrast": (0, 100),
            "sharpness": (0, 1000),
            "size": (0, 10000000),  # 10MB
            "resolution": (0, 4096 * 2160),  # 4K
            "color_balance": (0, 100)
        }
        
        if criterion in ranges:
            min_val, max_val = ranges[criterion]
            normalized = (value - min_val) / (max_val - min_val)
            return min(max(normalized, 0.0), 1.0)
        
        return 0.0

@dataclass
class Transaction:
    id: str
    state: TransactionState
    image_data: bytes
    metrics: dict = None
    votes: Dict[str, float] = None
    participating_nodes: Set[str] = None
    timestamp: float = None

    def __init__(self, id: str, image_data: bytes):
        self.id = id
        self.state = TransactionState.INIT
        self.image_data = image_data
        self.metrics = {}
        self.votes = {}
        self.participating_nodes = set()
        self.timestamp = time.time()

class MasterNode:
    def __init__(self, node_id: str, host: str, port: int):
        self.node_id = node_id
        self.transactions: Dict[str, Transaction] = {}
        self.worker_nodes: Dict[str, Tuple[str, int]] = {}
        self.wal = WriteAheadLog(f"master_{node_id}_wal.log")
        self.timeout = 5.0
        self.network = NetworkManager(host, port)
        self.collage_generator = CollageGenerator("output_collages")
        self.recovery_mode = False
        self.heartbeat_interval = 2.0
        self.node_health: Dict[str, float] = defaultdict(float)

    async def start(self):
        await self.network.start_server()
        await self.recover_from_log()
        await asyncio.gather(
            self.process_messages(),
            self.check_timeouts(),
            self.send_heartbeats()
        )

    async def send_heartbeats(self):
        while True:
            message = {
                "type": MessageType.HEARTBEAT.value,
                "sender": self.node_id,
                "timestamp": time.time()
            }
            
            for worker_id in self.worker_nodes:
                await self.network.send_message(worker_id, message)
            
            await asyncio.sleep(self.heartbeat_interval)

    async def handle_heartbeat_response(self, worker_id: str, timestamp: float):
        self.node_health[worker_id] = timestamp
        
        # Check for node recovery
        if worker_id not in self.worker_nodes:
            logger.info(f"Worker node {worker_id} has recovered")
            # Trigger recovery procedure
            await self.handle_node_recovery(worker_id)

    async def handle_node_recovery(self, worker_id: str):
        # Send current state to recovered node
        active_transactions = {
            tx_id: tx for tx_id, tx in self.transactions.items()
            if tx.state in [TransactionState.PREPARED, TransactionState.INIT]
        }
        
        for tx_id, transaction in active_transactions.items():
            message = {
                "type": MessageType.PREPARE.value,
                "transaction_id": tx_id,
                "sender": self.node_id,
                "data": {
                    "image_data": transaction.image_data,
                    "state": transaction.state.value
                }
            }
            await self.network.send_message(worker_id, message)

    async def create_collage(self, completed_transactions: List[str]) -> bytes:
        # Gather all committed images with their scores
        images_with_scores = []
        for tx_id in completed_transactions:
            tx = self.transactions.get(tx_id)
            if tx and tx.state == TransactionState.COMMITTED:
                # Calculate final score as average of votes
                score = sum(tx.votes.values()) / len(tx.votes) if tx.votes else 0
                images_with_scores.append((tx.image_data, score))

        # Generate collage using weighted layout
        return self.collage_generator.generate_collage(images_with_scores, "weighted")

    async def initiate_transaction(self, image_data: bytes) -> str:
        transaction_id = str(uuid.uuid4())
        transaction = Transaction(transaction_id, image_data)
        
        # Calculate image metrics
        metrics = ImageQualityMetrics.calculate_metrics(image_data)
        transaction.metrics = metrics
        
        self.transactions[transaction_id] = transaction
        self.wal.append(transaction)
        
        message = {
            "type": MessageType.PREPARE.value,
            "transaction_id": transaction_id,
            "sender": self.node_id,
            "data": {
                "image_data": image_data,
                "metrics": metrics
            }
        }
        
        for worker_id in self.worker_nodes:
            await self.network.send_message(worker_id, message)
        
        return transaction_id

class WorkerNode:
    def __init__(self, node_id: str, host: str, port: int):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.transactions: Dict[str, Transaction] = {}
        self.wal = WriteAheadLog(f"worker_{node_id}_wal.log")
        self.network = NetworkManager(host, port)
        self.voting_system = VotingSystem()
        self.recovery_mode = False
        self.last_heartbeat = 0
        self.master_node_id = None

    async def start(self):
        await self.network.start_server()
        await self.recover_from_log()
        await asyncio.gather(
            self.process_messages(),
            self.monitor_master_health()
        )

    async def monitor_master_health(self):
        while True:
            current_time = time.time()
            if self.last_heartbeat > 0 and current_time - self.last_heartbeat > 10.0:
                logger.warning("Master node appears to be down")
                # Implement master node failover logic here
                await self.handle_master_failure()
            await asyncio.sleep(1)

    async def handle_master_failure(self):
        # In a real implementation, this would involve leader election
        # For now, we'll just wait for master recovery
        self.recovery_mode = True
        logger.info("Waiting for master node recovery...")

    async def handle_prepare(self, message: dict):
        transaction_id = message["transaction_id"]
        image_data = message["data"]["image_data"]
        metrics = message["data"]["metrics"]
        
        # Create new transaction
        transaction = Transaction(transaction_id, image_data)
        transaction.metrics = metrics
        
        # Calculate vote base