import psutil
import torch
import GPUtil
import time
import logging

logging.basicConfig(level=logging.INFO)

class ResourceManager:
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.gpu_monitor_interval = 10

    def monitor_system(self):
        """ Continuously monitors system resources """
        while True:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent

            logging.info(f"CPU Usage: {cpu_usage}%")
            logging.info(f"Memory Usage: {memory_usage}%")
            logging.info(f"Disk Usage: {disk_usage}%")

            if self.gpu_available:
                self.monitor_gpu()

            time.sleep(self.gpu_monitor_interval)

    def monitor_gpu(self):
        """ Monitors GPU usage """
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            logging.info(f"GPU {gpu.id}: {gpu.name}")
            logging.info(f"  - Memory Usage: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
            logging.info(f"  - Load: {gpu.load * 100}%")

    def optimize_training(self):
        """ Adjusts training settings dynamically based on available resources """
        if psutil.virtual_memory().percent > 80:
            logging.warning("⚠️ High Memory Usage! Consider reducing batch size.")
        if psutil.cpu_percent() > 90:
            logging.warning("⚠️ High CPU Usage! Training may slow down.")

        if self.gpu_available:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                if gpu.memoryUtil > 0.8:
                    logging.warning(f"High GPU Memory Usage on {gpu.name}! Try using DeepSpeed.")

if __name__ == "__main__":
    resource_manager = ResourceManager()
    resource_manager.monitor_system()
