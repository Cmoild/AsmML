import numpy as np
from numpy.typing import NDArray
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms


class Linear:
    def __init__(self, in_features: int, out_features: int):
        rng = np.random.default_rng()
        self.weight: NDArray[np.float32] = rng.normal(
            0.0,
            np.sqrt(1.0 / in_features),
            size=(out_features, in_features),
        ).astype(np.float32)
        self.bias: NDArray[np.float32] = rng.normal(
            0.0,
            np.sqrt(1.0 / in_features),
            size=(out_features,),
        ).astype(np.float32)
        self.output: NDArray[np.float32] = np.array(None)
        self.input: NDArray[np.float32] = np.array(None)
        self.grad_input: NDArray[np.float32] = np.array(None)
        self.grad_output: NDArray[np.float32] = np.array(None)
        self.grad_weight: NDArray[np.float32] = np.array(None)
        self.grad_bias: NDArray[np.float32] = np.array(None)

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        self.input = x
        self.output = x @ self.weight.T + self.bias
        return self.output

    def update_grad(self, grad_input: NDArray[np.float32]) -> NDArray[np.float32]:
        self.grad_input = grad_input
        self.grad_output = self.grad_input @ self.weight
        self.grad_weight = self.grad_input.T @ self.input
        self.grad_bias = self.grad_input.sum(axis=0)
        return self.grad_output


class ReLU:
    def __init__(self):
        self.output: NDArray[np.float32] = np.array(None)
        self.input: NDArray[np.float32] = np.array(None)
        self.grad_input: NDArray[np.float32] = np.array(None)
        self.grad_output: NDArray[np.float32] = np.array(None)

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        self.input = x
        self.output = np.maximum(x, 0)
        return self.output

    def update_grad(self, grad_input: NDArray[np.float32]) -> NDArray[np.float32]:
        self.grad_input = grad_input
        self.grad_output = self.grad_input * (self.input > 0)
        return self.grad_output


class Softmax:
    def __init__(self):
        self.output: NDArray[np.float32] = np.array(None)
        self.input: NDArray[np.float32] = np.array(None)
        self.grad_input: NDArray[np.float32] = np.array(None)
        self.grad_output: NDArray[np.float32] = np.array(None)

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        self.input = x
        self.output = np.exp(x - x.max(axis=-1, keepdims=True)) / np.sum(
            np.exp(x - x.max(axis=-1, keepdims=True)), axis=-1, keepdims=True
        )
        return self.output


def compute_loss(
    true: NDArray[np.float32], pred: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    per_sample = -np.sum(true * np.log(pred), axis=-1)
    loss = per_sample.mean()
    grad_loss = (pred - true) / true.shape[0]
    return loss, grad_loss


transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(train_dataset, 50, True)


fc1 = Linear(28 * 28, 100)
fc2 = Linear(100, 10)
relu = ReLU()
softmax = Softmax()

epochs = 10
lr = np.float32(0.001)

for epoch in range(epochs):
    for i, (inp, outp) in enumerate(loader):
        B = inp.shape[0]
        inp = inp.reshape(B, 28 * 28)
        input: NDArray[np.float32] = inp.numpy()
        true: NDArray[np.float32] = np.eye(10, dtype=np.float32)[outp.numpy()]
        x = fc1.forward(input)
        x = relu.forward(x)
        x = fc2.forward(x)
        pred = softmax.forward(x)

        loss, grad_loss = compute_loss(true, pred)
        x = fc2.update_grad(grad_loss)
        x = relu.update_grad(x)
        x = fc1.update_grad(x)

        fc2.weight -= lr * fc2.grad_weight
        fc2.bias -= lr * fc2.grad_bias
        fc1.weight -= lr * fc1.grad_weight
        fc1.bias -= lr * fc1.grad_bias

        if i % 100 == 0:
            print(f"Epoch: {epoch} \t| Step: {i}/{len(loader)}  \t| Loss: {loss:.4f}")
