"""Train the Physics-Informed Attention U-Net on a preprocessed pickle dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from gas_source_navigation.gas import PhysicsInformedAttentionUNet
from gas_source_navigation.gas.data import GasFieldDataset
from gas_source_navigation.gas.losses import PhysicsInformedLoss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiments/pi_unet.pt"))
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument(
        "--no-data-loss",
        action="store_true",
        help="Monitor reconstruction MSE but exclude it from the optimized loss, as in the final notebook",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = GasFieldDataset(args.dataset)
    validation_size = max(1, round(len(dataset) * args.validation_ratio))
    train_size = len(dataset) - validation_size
    if train_size < 1:
        raise ValueError("Dataset must contain at least two samples")
    train_data, validation_data = random_split(
        dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsInformedAttentionUNet().to(device)
    criterion = PhysicsInformedLoss(
        reconstruction_weight=0.0 if args.no_data_loss else 1.0,
        physics_weight=args.physics_weight,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_validation = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = []
        for inputs, gas, sigma, position in train_loader:
            inputs, gas, sigma, position = (
                inputs.to(device),
                gas.to(device),
                sigma.to(device),
                position.to(device),
            )
            optimizer.zero_grad()
            terms = criterion(model(inputs), gas, sigma, position)
            terms.total.backward()
            optimizer.step()
            totals.append(terms.total.detach().item())

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for inputs, gas, _, _ in validation_loader:
                prediction = model(inputs.to(device))
                validation_losses.append(
                    torch.nn.functional.mse_loss(prediction.concentration, gas.to(device)).item()
                )
        validation_mse = sum(validation_losses) / len(validation_losses)
        if validation_mse < best_validation:
            best_validation = validation_mse
            torch.save(model.state_dict(), args.output)
        print(
            f"epoch={epoch:04d} train_loss={sum(totals) / len(totals):.6f} "
            f"validation_mse={validation_mse:.6f}"
        )


if __name__ == "__main__":
    main()
