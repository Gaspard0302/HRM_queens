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
