# mojo-mnist

MNIST MLP in PyTorch with optional Mojo kernels for inference. Includes a demo UI to draw digits and get predictions.

## Training

Run once to produce the weights file used by the demo:

```bash
pixi run python -m demo.mlp train
```

This exports the model to `.data/mlp_mnist.pth`.

For CNN:

```bash
pixi run python -m demo.cnn train
```

This exports the model to `.data/cnn_mnist.pth`.

## Demo

Start the server (serves the UI and the predict API):

```bash
pixi run python -m demo.server
```

Open http://127.0.0.1:8000/ and draw a digit.
