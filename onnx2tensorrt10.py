import tensorrt as trt

onnx_file_path = "efficientnet.onnx"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(TRT_LOGGER)

network = builder.create_network(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 if supported
config.set_memory_pool_limit(
    trt.MemoryPoolType.WORKSPACE, 1 << 32
)  # 4 GB (add if multiple engines are running)

with open(onnx_file_path, "rb") as model_file:
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(model_file.read()):
        print("Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()

input_tensor = network.get_input(0)  # Assuming the input tensor is at index 0
input_shape = input_tensor.shape
print("Model input shape:", input_shape)

# Create optimization profile for dynamic input shape
profile = builder.create_optimization_profile()
min_shape = (1, 3, 224, 224)  # Minimum shape (batch=1)
opt_shape = (1, 3, 224, 224)  # Optimal shape (batch=4)
max_shape = (1, 3, 224, 224)  # Maximum shape (batch=16)

profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
config.add_optimization_profile(profile)

engine_bytes = builder.build_serialized_network(network, config)

with open("efficientnet.engine", "wb") as f:
    f.write(engine_bytes)
