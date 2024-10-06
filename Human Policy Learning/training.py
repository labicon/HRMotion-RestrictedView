import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from diffusers.optimization import get_scheduler

def train_model(model, train_dataset, val_dataset, noise_scheduler, ema, model_path):
    """Train the model."""
    num_epochs = 100
    criterion = torch.nn.MSELoss()
    batch_size = 256
    accumulation_steps = 2
    early_stopping_patience = 20

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * num_epochs
    )

    scaler = GradScaler()
    writer = SummaryWriter()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for i, (inputs, actions) in enumerate(train_loader):
            inputs, actions = inputs.to(model.device), actions.to(model.device)

            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (inputs.size(0),)).to(model.device)
            noise = torch.randn_like(actions)
            noisy_actions = noise_scheduler.add_noise(actions, noise, t)

            location = inputs
            noisy_inputs = noisy_actions

            with autocast():
                noise_pred = model(noisy_inputs, t, location)
                loss = criterion(noise_pred, noise)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.step(model.parameters())
                lr_scheduler.step()

            train_loss += loss.item() * inputs.size(0)

        if (i + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.step(model.parameters())
            lr_scheduler.step()

        train_loss /= len(train_loader.dataset)
        writer.add_scalar('Loss/train', train_loss, epoch)

        val_loss = validate_model(model, val_loader, noise_scheduler, criterion)
        writer.add_scalar('Loss/val', val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, ema, optimizer, lr_scheduler, epoch, best_val_loss, model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping")
                break

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    writer.close()

def validate_model(model, val_loader, noise_scheduler, criterion):
    """Evaluate the model on the validation dataset."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, actions in val_loader:
            inputs, actions = inputs.to(model.device), actions.to(model.device)

            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (inputs.size(0),)).to(model.device)
            noise = torch.randn_like(actions)
            noisy_actions = noise_scheduler.add_noise(actions, noise, t)

            location = inputs
            noisy_inputs = noisy_actions

            with autocast():
                noise_pred = model(noisy_inputs, t, location)
                loss = criterion(noise_pred, noise)

            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    return val_loss

def save_checkpoint(model, ema, optimizer, lr_scheduler, epoch, best_val_loss, model_path):
    """Save the model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, model_path)