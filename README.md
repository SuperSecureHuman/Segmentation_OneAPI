# Segmentation with OpenVINO: Enhancing Semantic Segmentation

## Table of Contents

- [Introduction](#introduction)
- [Demo](#demo)
- [Model Weights](#model-weights)
- [Dataset](#dataset)
- [Methods](#methods)
- [Inference Performance](#inference-performance)
- [Integration of OneAPI](#integration-of-oneapi)
- [Other Integration Possiblities](#other-integration-possiblites)
- [Further Works](#further-works)
- [License](#license)

## Introduction

Semantic segmentation, a cornerstone of computer vision, involves understanding images at a pixel level. This repository embarks on a journey from training a robust semantic segmentation model on Oxford's Pets dataset to optimizing it for real-time inference using OpenVINO. The goal is to maintain exceptional accuracy while achieving lightning-fast performance.

## Demo

A short video demonstration showcases the efficiency and accuracy achieved through the integration of OpenVINO.

Run `demo.py`. Make sure to have model weights in the path `./model/model_int8.xml`



<https://github.com/SuperSecureHuman/Segmentation_OneAPI/assets/88489071/efe54327-e7c7-4e03-bf90-008340190f96>



## Model Weights

Model weights can be found in the Releases page. In order to run the final demo, you need only the INT8 files.

## Dataset

I used the [Oxford's Pets dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) for our training and evaluation. This dataset offers a rich collection of diverse pet images along with meticulous pixel-wise annotations, providing the foundation for seamless semantic segmentation tasks.

## Project Stack

1. Pytorch
2. OpenCV
3. Pytorch Segmentation Models
4. NNCF (Neural Network Compression Framework for enhanced OpenVINO™ inference)
5. OpenVINO
6. Intel Extention for Pytorch

## Methods

The evolution of our model is outlined through these pivotal methods:

1. **Torch**: Training of the model. And using torch for initial baseline to start with.
2. **TorchScript**: Transitioning to TorchScript enables dynamic Just-In-Time (JIT) compilation, enhancing computational efficiency.
3. **Intel PyTorch Extension (IPEX) - BFFloat**: IPEX with BFFloat optimization leverages oneDNN. IPEX applies graph fusion, which is accelerated by oneDNN.
4. **Testing Quantization**: Quantization with Callibration data was used then to compress the model to INT8 without compermising on accuracy.
5. **OpenVINO Integration**: The optimized model seamlessly converts to OpenVINO's Intermediate Representation (IR) format, delivering fast inference speed.

## Inference Performance

The leap in inference performance is astonishing:

- Initial PyTorch CPU inference speed: ~16 frames per second (fps).
- Inference speed with OpenVINO CPU (INT8 precision): ~100 fps.

Notably, this performance enhancement is achieved without any compromise on segmentation accuracy.

## Integration of OneAPI

Optimization is at the heart of this project, facilitated by OneAPI tools:

1. **Export to ONNX**: Model Optimizer (MO) exports the model to ONNX format for compatibility with OpenVINO.
2. **NNCF Quantization**: Neural Network Compression Framework (NNCF) enables INT8 quantization and exporting.
3. **Intel OpenVINO Inference Engine**: The Intel OpenVINO Inference Engine drives rapid execution during inference.

### Other Integration Possiblites

1. Can use other Intel Hardware (Intel Xeon Processors, Habana Gaudi Instances, Intel Server GPUs) via DevCloud for training by leveraging IPEX and Torch XPU interface. Since I had a GPU locally, I preffered using it.
2. Host an async inference server, and use the model for real-time inference with data from other devices. Couldn't do because of lack of hardware presently with me.

## Further Works

1. Use the model for real-time inference with data from other devices (Make use of inference server within OpenVINO).
2. Extend this to use Intel NCS on edge Devices

## License

This project operates under the [MIT License](LICENSE), granting you the freedom to manipulate, adjust, and share the code while adhering to the original license terms.

---
