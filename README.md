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
Conditional Encoding Architecture

<img src="/figures/2_page-0001.jpg" alt="图片alt" title="Conditional Diffusion Architecture">
Conditional Diffusion Architecture

## GIF 效果展示

以下展示了两组 GIF 图片，每组包含两个部分。每个 GIF 的左侧显示输入条件的标签（1p~5p），右侧显示预测结果（0~15）。 

### 江苏模型

- **图片标题**: Jiangsu Model  
  ![Fig jiangsu model](/figures/videos_pred_0_0.gif)

- **图片标题**: Jiangsu Model (Another Scenario)  
  ![Fig jiangsu model](/figures/videos_pred_860000_0.gif)

### 全国模型

- **图片标题**: Nationwide Model  
  ![Fig nationwide model](/figures/nationwide_model.gif)

### 青海模型

- **图片标题**: Qinghai Model  
  ![Fig qinghai model](/figures/qinghai_model.gif)

- **图片标题**: Qinghai Model (Another Scenario)  
  ![Fig qinghai model](/figures/qinghai_model2.gif)

### 三亚模型

- **图片标题**: Sanya Model  
  ![Fig sanya model](/figures/sanya_model.gif)

## Experimental Comparison
<img src="/figures/3_page-0001.jpg" alt="图片alt" title="Conditional Diffusion Architecture">

<img src="/figures/4_page-0001.jpg" alt="图片alt" title="Conditional Diffusion Architecture">

## Train and Test
`CUDA_VISIBLE_DEVICES=3,4 python runner.py --config configs/weather_round.yml --exp weather_20  --config_mod sampling.subsample=100 -t --ni`

`CUDA_VISIBLE_DEVICES=1 python runner.py --config configs/weather_round.yml --exp weather_20  --config_mod sampling.subsample=100 sampling.num_frames_pred=100 data.revise=False -vg --ni`
