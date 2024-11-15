import torch
import onnx

from model import create_model


def export_to_onnx():
    # Create model
    model = create_model()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "efficientnet.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )

    # Verify ONNX model
    onnx_model = onnx.load("efficientnet.onnx")
    onnx.checker.check_model(onnx_model)
    print("ONNX export successful")


if __name__ == "__main__":
    export_to_onnx()
