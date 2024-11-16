import torch
import onnx
import onnx_graphsurgeon as gs
from model import create_model


def export_to_onnx_explicit_batch():
    # Create model
    model = create_model()

    # Create dummy input with explicit batch size
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    # Export to ONNX with explicit batch size
    onnx_path = "efficientnet.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,  # Use ONNX opset version 11
    )

    # Load the ONNX model
    onnx_model = onnx.load(onnx_path)

    # Modify the graph to enforce explicit batch size using ONNX GraphSurgeon
    graph = gs.import_onnx(onnx_model)
    for tensor in graph.tensors().values():
        if (
            tensor.shape and tensor.shape[0] == "batch_size"
        ):  # Replace symbolic batch size with a fixed size
            tensor.shape[0] = batch_size

    # Export the modified ONNX model
    graph.cleanup().toposort()
    onnx_model_explicit = gs.export_onnx(graph)

    # Save and verify the modified model
    modified_onnx_path = "efficientnet.onnx"
    onnx.save(onnx_model_explicit, modified_onnx_path)
    onnx.checker.check_model(onnx_model_explicit)

    print(
        f"ONNX export successful with explicit batch size. Saved as {modified_onnx_path}"
    )


if __name__ == "__main__":
    export_to_onnx_explicit_batch()
