# Segmentation with OpenVINO: Enhancing Semantic Segmentation

Welcome to the Segmentation with OpenVINO repository! Here, we dive into the captivating world of semantic segmentation, showcasing the journey from training a segmentation model on the Oxford's Pets dataset to optimizing it for lightning-fast inference using OpenVINO. This repository demonstrates how the strategic integration of OpenVINO can significantly accelerate inference while maintaining remarkable accuracy.

## Introduction

The main goal of this repository is to guide you through the process of training a robust semantic segmentation model on the Oxford's Pets dataset. However, the journey doesn't stop there – we take the model's performance to the next level by meticulously optimizing it for rapid inference. This readme serves as your compass, helping you navigate through the tools, techniques, and transformations that breathe life into the model, all efficiently powered by OpenVINO.

## Dataset

The foundation of this endeavor is the esteemed [Oxford's Pets dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). This comprehensive dataset provides a rich collection of diverse pet images, accompanied by detailed pixel-wise annotations essential for seamless semantic segmentation tasks.

## Evolutionary Methods

This repository unveils the story of the model's refinement through various pivotal methods:

- **Torch:** The journey commences with PyTorch, laying the groundwork for subsequent optimizations.

- **TorchScript:** Witness the metamorphosis as the model transitions into TorchScript, enabling dynamic Just-In-Time (JIT) compilation for enhanced efficiency.

- **Intel PyTorch Extension (IPEX) - BFFloat:** Unleash the power of IPEX with BFFloat optimization, propelling the model's computation into overdrive.

- **OpenVINO:** The pièce de résistance – observe as the optimized PyTorch model seamlessly transforms into OpenVINO's Intermediate Representation (IR) format, elevating inference speed to unprecedented levels.

## Inference Improvements

Prepare to be astonished by the achievement of boosting inference performance:

- Initial PyTorch CPU inference speed: ~16 frames per second (fps).
- Inference speed with OpenVINO CPU (INT8 precision): ~100 fps.
- It's worth noting that this speed surge is accomplished without compromising segmentation accuracy.

## Progression

Journey, witnessing the model's speed surge from ~16 fps to ~100 fps using OpenVINO:

- Initial steps delve into Torch.
- JIT and IPEX come next, with quantization playing a minor role.
- A turning point arrives with nncf and a calibration dataset. The result? INT8 quantization, delivering staggering performance.

## License to Thrive

This project operates under the umbrella of the [MIT License](LICENSE), granting you the freedom to manipulate, adjust, and share the code while preserving the original license terms.

---

Explore the repository's directories to unravel the array of optimization methods. Witness the transformation of the model from its nascent state to a dynamically optimized, OpenVINO-fueled masterpiece. If questions or ideas arise, don't hesitate to reach out – the door is always open for connection.
