import wandb
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize WandB
wandb.init(project="llama-monitoring")

def monitor_training():
    """
    Monitors training logs from Weights & Biases and sends alerts if loss is too high.
    """
    api = wandb.Api()
    project_name = "AssignmentXNL/llama-monitoring"

    while True:
        try:
            runs = api.runs(project_name)

            if runs:
                latest_run = runs[0]
                summary = latest_run.summary

                loss = summary.get("train/loss", None)

                if loss is not None:
                    logging.info(f"Monitoring: Latest loss = {loss:.4f}")

                    if loss > 1.5:
                        wandb.alert(
                            title="High Training Loss",
                            text=f"Loss Alert! Current Loss: {loss:.4f}",
                        )
                        logging.warning(f"High loss detected: {loss:.4f}")

                else:
                    logging.info("Loss value not found in summary.")

            else:
                logging.info("No active runs found.")

        except Exception as e:
            logging.error(f"Error fetching W&B runs: {e}")

        time.sleep(300)

if __name__ == "__main__":
    monitor_training()
