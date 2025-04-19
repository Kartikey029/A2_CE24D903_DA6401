# iNaturalistCNN 

This project fine-tunes a custom convolutional neural network (CNN), **iNaturalistCNN**, on the [nature_12K](https://www.kaggle.com/datasets/jpullen/nature12) dataset (specifically, the iNaturalist subset). Training is implemented using PyTorch Lightning and tracked with Weights & Biases (wandb), including experiment logging and hyperparameter sweeping.

---

## 📦 Features

- Custom CNN with 5 convolutional blocks
- Multiple activation functions: `relu`, `gelu`, `silu`, `mish`, `none`, `sigmoid`, `tanh`
- Filter organization strategies: `constant`, `doubling`, `halving`
- Support for dropout and batch normalization
- Stratified dataset splitting for balanced class distribution
- Optional data augmentation
- Integration with wandb for visualization and logging
- Early stopping to prevent overfitting

---

## 🧠 Model Definition

The model, `iNaturalistCNN`, inherits from `pl.LightningModule` and has the following configurable structure:

### Key Parameters

- `conv_configs`: List of tuples defining convolutional layers `(filters, kernel size)`
- `dense_neurons`: Number of neurons in the fully connected layer
- `dropout_rate`: Dropout probability
- `activation_fn`: Activation function for both convolutional and dense layers
- `use_batch_norm`: Boolean to toggle batch normalization
- `learning_rate`: Learning rate for the optimizer
- `input_size`: Size of input images
- `num_classes`: Number of target classes

### Functions in `iNaturalistCNN`

```python
class iNaturalistCNN(pl.LightningModule):
    def __init__(self, conv_configs, dense_neurons, dropout_rate, conv_activation,
                 dense_activation, use_batch_norm, learning_rate, input_size, num_classes):
        # Initialize model layers and parameters

    def forward(self, x):
        # Forward pass through the network

    def training_step(self, batch, batch_idx):
        # Training loop logic (single batch)

    def validation_step(self, batch, batch_idx):
        # Validation loop logic (single batch)

    def test_step(self, batch, batch_idx):
        # Testing loop logic (single batch)

    def configure_optimizers(self):
        # Optimizer and learning rate scheduler configuration
```

---

## 🏃‍♂️ Training Function

The primary training function, `run_training()`, sets up hyperparameters from wandb, prepares the data, initializes the model, and configures the PyTorch Lightning trainer.

### Function Signature

```python
def run_training():
```

This function retrieves all training parameters via `wandb.config`.

### Hyperparameters (via wandb.config)

- `train_data_path`: Directory path for training data
- `validation_data_path`: Directory path for validation data
- `learning_rate`: Optimizer learning rate
- `batch_size`: Batch size for the DataLoader
- `dense_neurons`: Neurons in the dense layer
- `dropout_rate`: Dropout same for the FCN and CNN
- `conv_activation`: Activation for convolutional layers
- `dense_activation`: Activation for the dense layer
- `batch_norm`: Enable/disable batch normalization
- `filter_organization`: Filter pattern (`constant`, `doubling`, `halving`)
- `constant_filter` : All layers get the same number of filters.
- `base_filter`: Filters increase/decrease across layer based on the what filter organization is there except "Constant". 
- `data_augmentation`: Enable/disable augmentation 
- `weight_decay`: Optimizer weight decay
- `project_name`: wandb project identifier

---

## 🧪 `run_training` Function Breakdown

```python
def run_training():
    wandb.init()
    config = wandb.config

    # Configure convolutional filter layout
    filt_org = config.get("filter_organization", "constant")
    if filt_org == "constant":
        conv_configs = [(config.constant_filter, 3)] * 5
    elif filt_org == "doubling":
        base = config.base_filter
        conv_configs = [(base * (2 ** i), 3) for i in range(5)]
    elif filt_org == "halving":
        base = config.base_filter
        conv_configs = [(base // (2 ** i), 3) for i in range(5)]
    else:
        raise ValueError("Unknown filter organization!")

    # Apply data augmentation
    use_data_aug = config.get("data_augmentation", True)
    transform_train, transform_test = get_transforms(use_data_aug)

    # Load datasets
    train_dataset_full = ImageFolder(root=config.train_data_path, transform=transform_train)
    train_idx, test_idx = get_train_test_split(train_dataset_full)
    train_dataset = Subset(train_dataset_full, train_idx)
    test_dataset = Subset(train_dataset_full, test_idx)
    val_dataset = ImageFolder(root=config.validation_data_path, transform=transform_test)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=int(config.batch_size), shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=int(config.batch_size), shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=int(config.batch_size), shuffle=False, num_workers=4)

    # Instantiate model
    model = iNaturalistCNN(
        num_classes=10,
        learning_rate=config.learning_rate,
        input_size=128,
        conv_configs=conv_configs,
        dense_neurons=int(config.dense_neurons),
        dropout_rate=config.dropout_rate,
        conv_activation=config.conv_activation,
        dense_activation=config.dense_activation,
        use_batch_norm=config.batch_norm
    )

    # Setup wandb logger and callbacks
    wandb_logger = WandbLogger(project=config.project_name)
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=True)

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[early_stop_callback]
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, dataloaders=test_loader)
    wandb.finish()
```

---

## 🚀 Running the Sweep

```python
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project=sweep_config["parameters"]["project_name"]["value"])
    wandb.agent(sweep_id, function=run_training)
```

---

## 📊 Sweep Configuration

- Grid search across activations, dropout rates, weight decay, and more
- Configurable via Python dictionary
- Includes batch size, learning rate, augmentation toggle, etc.

---

## 📌 Requirements

```bash
pytorch-lightning
wandb
scikit-learn
torchvision
matplotlib
```

