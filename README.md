# OnnxruntimeDetector
onnxruntime wrapper
=======
# 使用Onnxruntime 的YOLOv5 模型推理示例

## 简介

这个仓库包含了一个用于执行YOLOv5模型推理的示例代码。YOLOv5是一种流行的目标检测模型，能够在图像中检测多个对象。

## 使用

要使用此示例，你需要安装以下依赖项：

- OnnxRuntime：用于加载和运行ONNX模型。
- OpenCV：用于图像处理。
- 编译工具（如CMake和Make）。

执行以下步骤：

1. 克隆此仓库：

   ```bash
   git clone -b master https://github.com/wxxz975/OnnxruntimeDetector.git
   ```

2. 进入项目目录:

   ```bash
   cd OnnxruntimeDetector
   ```

3. 创建一个build目录并进入：
    ```bash
    mkdir build && cd build
    ```

4. 使用CMake构建项目并且生成可执行文件：
     ```bash
        cmake .. && make -j4
    ```
