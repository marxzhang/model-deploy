import os
import tensorrt as trt


def build_engine(onnx_path, engine_path):
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        assert parser.parse(f.read()), "failed to parse ONNX: {}".format(
            "; ".join(str(e) for e in parser.get_error(30) if e.code != -1))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.INT8)

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    shape = tuple(network.get_input(0).shape)
    profile.set_shape(input_name, min=shape, opt=shape, max=shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None, "engine build failed"
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print("engine saved to", engine_path)


if __name__ == "__main__":
    build_engine("resnet34.onnx", "trt_output/resnet34.trt")