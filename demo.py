import asyncio
import logging
import os
from PIL import Image
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CollageNode:
    def __init__(self, node_id: str, input_dir: str, output_dir: str):
        self.node_id = node_id
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.images = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    async def load_images(self):
        """Load images from the input directory"""
        logger.info(f"Node {self.node_id}: Loading images from {self.input_dir}")
        
        for file_name in os.listdir(self.input_dir):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    image_path = os.path.join(self.input_dir, file_name)
                    image = Image.open(image_path)
                    self.images[file_name] = image
                    logger.info(f"Loaded image: {file_name}")
                except Exception as e:
                    logger.error(f"Failed to load {file_name}: {str(e)}")
        
        return len(self.images)

    def calculate_image_score(self, image):
        """Calculate a simple score based on image characteristics"""
        # This is a simplified scoring system - you can make it more sophisticated
        width, height = image.size
        aspect_ratio = width / height
        
        # Prefer images closer to standard aspect ratios
        aspect_score = min(1.0, abs(aspect_ratio - 1.5))
        
        # Add more scoring criteria here as needed
        return aspect_score

    async def generate_collage(self):
        """Generate a collage from loaded images"""
        if not self.images:
            raise ValueError("No images loaded")

        logger.info("Generating collage...")

        # Sort images by score
        scored_images = [
            (img_name, self.calculate_image_score(img))
            for img_name, img in self.images.items()
        ]
        scored_images.sort(key=lambda x: x[1], reverse=True)

        # Calculate collage dimensions
        n_images = len(self.images)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
        
        # Size for each image in the collage
        thumb_width = 300
        thumb_height = 200
        
        # Create new image with white background
        collage = Image.new('RGB', 
                          (thumb_width * cols, thumb_height * rows), 
                          'white')

        # Place images in the collage
        for idx, (img_name, _) in enumerate(scored_images):
            img = self.images[img_name]
            
            # Resize image maintaining aspect ratio
            img.thumbnail((thumb_width, thumb_height))
            
            # Calculate position
            row = idx // cols
            col = idx % cols
            x = col * thumb_width
            y = row * thumb_height
            
            # Center image in its slot
            x_offset = (thumb_width - img.width) // 2
            y_offset = (thumb_height - img.height) // 2
            
            collage.paste(img, (x + x_offset, y + y_offset))

        # Save collage
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"collage_{timestamp}.jpg")
        collage.save(output_path, quality=95)
        logger.info(f"Collage saved to: {output_path}")
        return output_path

async def main():
    # Set up paths
    base_dir = Path(__file__).parent
    input_dir = base_dir / "images"
    output_dir = base_dir / "output"
    
    # Create directories if they don't exist
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Check if there are images in the input directory
    if not any(input_dir.glob('*.jpg')) and not any(input_dir.glob('*.png')):
        logger.error(f"No images found in {input_dir}. Please add some images first!")
        return
    
    # Create and run node
    node = CollageNode("demo_node", str(input_dir), str(output_dir))
    
    # Load images
    image_count = await node.load_images()
    logger.info(f"Loaded {image_count} images")
    
    if image_count > 0:
        # Generate collage
        output_path = await node.generate_collage()
        logger.info(f"Process completed! Collage saved to: {output_path}")
    else:
        logger.error("No images were loaded successfully")

if __name__ == "__main__":
    asyncio.run(main())