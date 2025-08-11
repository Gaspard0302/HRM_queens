# Hierarchical Reasoning Model (MLX Swift)

This repo implements HRM w/ ACT for Sudoku puzzles in Swift using the MLX framework.

## Build & Run

To build the code, you may either use Xcode directly, or use the following `xcodebuild` command:

```
xcodebuild build -scheme HierarchicalReasoningModel -configuration Release -destination 'platform=OS X' -derivedDataPath ./build
```

Then, to run the compiled binary, you can invoke the following command to train the model:

```
./build/Build/Products/Release/HierarchicalReasoningModel train
```

Or you can invoke the following command to run inference for a single random puzzle using the given checkpoint at a certain difficulty level:

```
./build/Build/Products/Release/HierarchicalReasoningModel infer <checkpoint path> <difficulty>
```

## Citation

If you use this repo, please cite:

Wang, G., Li, J., Sun, Y., Chen, X., Liu, C., Wu, Y., Lu, M., Song, S., & Abbasi Yadkori, Y. (2025). *Hierarchical Reasoning Model*. arXiv. https://doi.org/10.48550/arXiv.2506.21734

```bibtex
@misc{wang2025hrm,
  title         = {Hierarchical Reasoning Model},
  author        = {Wang, Guan and Li, Jin and Sun, Yuhao and Chen, Xing and Liu, Changling and Wu, Yue and Lu, Meng and Song, Sen and Abbasi Yadkori, Yasin},
  year          = {2025},
  eprint        = {2506.21734},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2506.21734},
  url           = {https://arxiv.org/abs/2506.21734}
}
```
