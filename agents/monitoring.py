import wandb
import time

wandb.init(project="llama-monitoring")

def monitor_training():
    while True:
        logs = wandb.Api().runs(path="llama-monitoring")
        latest_run = logs[0]

        loss = latest_run.summary.get("train/loss", None)
        if loss is not None:
            if loss > 1.5:
                wandb.alert(
                    title="Training Issue Detected",
                    text=f"High loss detected: {loss}",
                )
        
        time.sleep(300)

if __name__ == "__main__":
    monitor_training()
