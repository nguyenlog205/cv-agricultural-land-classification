import torch

# Giả định các hàm này đã được định nghĩa trong các file tương ứng
from utils import load_config
from data_module import create_dataloaders 
from model import LandClassifierModel


def train(
    train_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    num_epochs,
    device,
    checkpoint_path="best_model.pth"
):
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }
    model.to(device)
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_predictions / total_predictions
        
        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_accuracy)


        val_accuracy = evaluate(model, val_loader, device)
        history["val_accuracy"].append(val_accuracy) # Ghi lại accuracy

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {epoch_loss:.4f}, "
              f"Train Accuracy: {epoch_accuracy:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print("Best model saved.")

    print("Training complete.")
    return model, history 

def evaluate(model, data_loader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    accuracy = correct_predictions / total_predictions
    return accuracy 

# ==============================================================
# CONFIG LOADERS
# ==============================================================
train_config = load_config("config/train_config.yml")
model_config = load_config("config/model_config.yml")
data_config  = load_config("config/data_config.yml")

# ==============================================================
# TRAINING PIPELINE
# ==============================================================
def training_pipeline(
    data_config,
    model_config,
    training_config
):
    

    # --- Create DataLoaders ---
    train_loader, val_loader, test_loader = create_dataloaders(
        data_config,
        batch_size=training_config['batch_size']
    )

    # --- Initialize Model ---
    model = LandClassifierModel(
        num_classes=model_config['num_classes'],
        freeze_features=model_config['freeze_features']
    )

    # --- Define Loss and Optimizer ---
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=training_config['learning_rate']
    )

    # --- Determine Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Train the Model ---
    trained_model, history = train(
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        training_config['num_epochs'],
        device
    )

    # --- Evaluate on Test Set ---
    test_accuracy = evaluate(trained_model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")

# ==============================================================
# MAIN EXECUTION BLOCK
# ==============================================================
if __name__ == "__main__":
    print("--- Starting Training Pipeline ---")
    # Kiểm tra xem PyTorch có nhận GPU không (dựa vào fix lỗi môi trường trước đó)
    if not torch.cuda.is_available():
        print("⚠️ Warning: CUDA device not found. Training on CPU.")
        
    try:
        # Gọi hàm chính để bắt đầu quá trình huấn luyện
        training_pipeline(
            data_config,
            model_config,
            train_config 
        )
    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}")