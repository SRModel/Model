# Model
Code is about to be released.


### Abstract
In single-image super-resolution reconstruction, existing deep learning-based methods often suffer from poor generalization performance due to the domain discrepancy between the training data and the real-world data. In this paper, we consider the domain disparity as a perturbation in the target data. Based on this idea, we propose a bi-level attack-embedded perturbation defense network for real-world super-resolution reconstruction. The devised network mainly comprises an attack layer, a detail-preserving defense layer and a reconstruction layer. The attack layer injects attack information into image features from two levels. One is on the high-frequency component and the other is on the pixel of the image. On the level of the high-frequency component, the attack is implemented by adjusting the mean and variance of this component to replicate the perturbation on high-frequency information. Meanwhile, the injection of attack at pixel simulates the perturbation in the image pixel caused by unidentified factors. In the defense layer, a bidirectional distillation mechanism is employed to maintain the coherence of the perturbed image's features with those of the unaltered image. This ensures that the network can deliver robust outcomes, even when the features have been subject to perturbations. Furthermore, a detail recovery component is integrated into the defense layer to address the detail information loss arising from the injection of attack information. The final reconstruction result is output by the reconstruction layer. The experimental results on three real-world datasets demonstrate the effectiveness of the proposed method and perform the superiority over the state-of-the-art approaches.

- Install dependencies. (Python 3.9 + NVIDIA GPU + CUDA. Recommend to use Anaconda)
```bash
pip install -r requirements.txt
```
- datasets
```bash
    ../DF2K+DIV8K+Flickr2K+OutdoorSceneTraining/train_HR_sub
    ../DF2K+DIV8K+Flickr2K+OutdoorSceneTraining/train_EDGE_sub
    ../DF2K+DIV8K+Flickr2K+OutdoorSceneTraining/meta_info_DF2K+DIV8.txt
```





