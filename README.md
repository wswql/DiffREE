# Extrapolation Algorithms for Radar Echoes Based on Conditional Diffusion Models
Conditional Diffusion Models

**Abstract:** The current inaccurate radar extrapolation results exhibit two typical features: the extension 
of the extrapolation time leads to increasingly attenuated echo strength, and the prediction 
performance for strong echoes decreases rapidly. This paper presents a Diffuse Radar Extrapolation 
Algorithm Driven by Radar Echo Frames (DiffREE) to address these issues. The algorithm employs a 
conditional coding module to deeply fuse past radar echo frames’ spatial and temporal information.
Additionally, it automatically extracts spatiotemporal features of the echoes through the Transformer 
encoder. These extracted features serve as the condition for the conditional diffusion model, which, in 
turn, drives the diffusion model to reconstruct the current radar echo frames. The experimental results 
demonstrate that the method can generate high-precision and high-quality radar forecast frames. 
Compared with the best baseline algorithm, the proposed method shows significant improvements
of 42.2%, 51.1%, 49.8%, and 39.5% in CSI, ETS, HSS, and POD metrics, respectively

## Model Structure
<img src="/figures/1_page-0001.jpg" alt="图片alt" title="Conditional Encoding Architecture">

<img src="/figures/2_page-0001.jpg" alt="图片alt" title="Conditional Diffusion Architecture">
## GIF Effect
Two groups of two with labels on the left and forecasts on the right, 1p\~5p in the upper left corner for input condition frames and 0\~15 for forecast frames
<img src="/figures/videos_pred_0_0.gif" alt="图片alt" title="图片title">
<img src="/figures/videos_pred_860000_0.gif" alt="图片alt" title="图片title">
