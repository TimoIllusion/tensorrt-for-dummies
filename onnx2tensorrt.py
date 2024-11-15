import tensorrt as trt


def build_engine():
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open("efficientnet.onnx", "rb") as model:
        parser.parse(model.read())

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    config.set_flag(trt.BuilderFlag.FP16)

    # Build engine
    engine = builder.build_engine(network, config)

    # Save engine
    with open("efficientnet.engine", "wb") as f:
        f.write(engine.serialize())


if __name__ == "__main__":
    build_engine()
