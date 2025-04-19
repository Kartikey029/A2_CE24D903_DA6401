# iNaturalistCNN Hyperparameter Sweep

This project fine-tunes a custom convolutional neural network (CNN) called **iNaturalistCNN** on the [nature\_12K](https://www.kaggle.com/datasets/jpullen/nature12) dataset (specifically the iNaturalist subset). The training uses PyTorch Lightning and Weights & Biases (wandb) for experiment tracking, model logging, and hyperparameter sweeping.



## 📦 Features

- Custom CNN with 5 convolutional blocks.
- Multiple activation function choices: `relu`, `gelu`, `silu`, `mish`, `none`, **Sigmoid, Tanh**
- Support for filter organization patterns: `constant`, `doubling`, `halving`.
- Dropout & Batch Normalization support.
- Stratified dataset splitting to ensure class balance.
- Built-in support for data augmentation.
- Integrated with wandb for training and sweep visualization.
- Early stopping for regularization.

---

## 🧠 Model Definition

The model is defined as `iNaturalistCNN`, inheriting from `pl.LightningModule` with the following structure:

### Key Parameters

- `conv_configs`: List of tuples defining convolution layers (filters, kernel size).
- `dense_neurons`: Number of neurons in the fully connected layer.
- `dropout_rate`: Dropout probability.
- `activation_fn`: Configurable activation for conv and dense layers.

---

## 🏃‍♂️ Training Function

The main training function `run_training()` handles the sweep-compatible configuration, model instantiation, data preprocessing, and trainer setup.

### Function Signature

```python
def run_training():
```

This function expects hyperparameters to be passed via `wandb.config`.

### Function Arguments via wandb.config

- `train_data_path`: Path to training data directory.
- `validation_data_path`: Path to validation data directory.
- `learning_rate`: Learning rate for optimizer.
- `batch_size`: Batch size for data loaders.
- `dense_neurons`: Number of neurons in the dense layer.
- `dropout_rate`: Dropout rate to apply (applies to both CNN and FCN).
- `conv_activation`: Activation function for conv layers.
- `dense_activation`: Activation function for dense layer.
- `batch_norm`: Boolean to use batch normalization.
- `filter_organization`: Strategy for convolutional filters (`constant`, `doubling`, `halving`).
- `constant_filter` or `base_filter`: Base filter value depending on strategy.
- `data_augmentation`: Toggle for using data augmentation.
- `weight_decay`: Regularization term for optimizer.
- `project_name`: wandb project name.

---

## 🧪 Function: run\_training

```python
def run_training():
    wandb.init()
    config = wandb.config

    # Filter organization
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

    # Data augmentation
    use_data_aug = config.get("data_augmentation", True)
    transform_train, transform_test = get_transforms(use_data_aug)

    # Load datasets
    train_dataset_full = ImageFolder(root=config.train_data_path, transform=transform_train)
    train_idx, test_idx = get_train_test_split(train_dataset_full)
    train_dataset = Subset(train_dataset_full, train_idx)
    test_dataset = Subset(train_dataset_full, test_idx)
    val_dataset = ImageFolder(root=config.validation_data_path, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=int(config.batch_size), shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=int(config.batch_size), shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=int(config.batch_size), shuffle=False, num_workers=4)

    # Create model
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

    # Logger and early stopping
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

- Grid search over activations, dropout, weight decay, and more.
- Configurable via Python dictionary (no YAML required).
- Includes batch size, learning rate, augmentation toggle, etc.

---

## 📌 Requirements

```
pytorch-lightning
wandb
scikit-learn
torchvision
matplotlib
```


