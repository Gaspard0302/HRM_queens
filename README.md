# Hierarchical Reasoning Model for Queens Game

A Swift implementation of a Hierarchical Reasoning Model (HRM) with Adaptive Computation Time (ACT), adapted from Sudoku solving to solving the Queens game puzzle.

## Overview

This repository contains a modified version of the original HRM that was designed for Sudoku solving, now repurposed to solve the Queens game. The Queens game is a constraint satisfaction puzzle where players must place exactly one queen in each row, column, and colored region of the board, while ensuring no two queens are diagonally adjacent (touching at corners).

## Key Modifications from Original Sudoku Implementation

### 1. **Data Source**
- **Original**: Generated Sudoku puzzles with varying difficulties
- **Modified**: Real Queens puzzles from [samimsu/queens-game-linkedin](https://github.com/samimsu/queens-game-linkedin)
- **Dataset**: 481 Queens levels downloaded, with 174 levels in 8x8 size (our primary training size)

### 2. **Model Architecture Changes**
- **Output Space**: Changed from 10 classes (digits 0-9) to 2 classes (binary: queen/no queen)
- **Sequence Length**: Adapted from 81 (9x9 Sudoku) to 64 (8x8 Queens)
- **Loss Function**: Switched from categorical cross-entropy to binary cross-entropy
- **Prediction**: Binary classification per cell instead of multi-class digit prediction

### 3. **Data Augmentation**
The system implements 8x data augmentation through geometric transformations:
- 4 rotations (0°, 90°, 180°, 270°)
- 4 reflections (horizontal, vertical, diagonal, anti-diagonal)

This transforms 174 base levels into 1,392 training examples for 8x8 boards.

### 4. **Game Logic Implementation**
Complete Queens validation system including:
- One queen per row/column/region constraints
- Diagonal adjacency checking (no queens touching at corners)
- Backtracking solver for generating solutions
- Win condition verification

## File Structure

```
Sources/HierarchicalReasoningModel/
├── Queens.swift                    # Queens game logic, validation, and data structures
├── QueensTraining.swift            # Queens-specific training system
├── HierarchicalReasoningModel.swift # Main entry point with Queens modes
├── Modelling/                      # Neural network architecture (unchanged)
├── Training.swift                  # Original Sudoku training (preserved)
└── Sudoku.swift                    # Original Sudoku logic (preserved)

queens_levels/                       # Downloaded Queens puzzle data
├── level1.ts
├── level2.ts
└── ... (481 levels total)
```

## Build & Run

### Build
You may either use Xcode directly, or use the following commands:

```bash
# Using xcodebuild
xcodebuild build -scheme HierarchicalReasoningModel -configuration Release -destination 'platform=OS X' -derivedDataPath ./build

# Or using Swift Package Manager
swift build -c release
```

### Training

#### Queens Training (New)
```bash
# Using xcodebuild output
./build/Build/Products/Release/HierarchicalReasoningModel train-queens

# Or using SPM
swift run HierarchicalReasoningModel train-queens
```

This will:
- Load 174 8x8 Queens levels
- Apply 8x data augmentation (1,392 total examples)
- Train the model with binary classification
- Save checkpoints as `queens-8x8-checkpoint-{step}.safetensors`

#### Original Sudoku Training (Preserved)
```bash
./build/Build/Products/Release/HierarchicalReasoningModel train
```

### Inference

#### Queens Inference
```bash
./build/Build/Products/Release/HierarchicalReasoningModel infer-queens <checkpoint-path> <level-file>

# Example:
./build/Build/Products/Release/HierarchicalReasoningModel infer-queens queens-8x8-checkpoint-250.safetensors queens_levels/level1.ts
```

#### Original Sudoku Inference
```bash
./build/Build/Products/Release/HierarchicalReasoningModel infer <checkpoint-path> <difficulty>
```

## Model Details

### Architecture
- **Hierarchical Reasoning**: Two-level reasoning system (high-level and low-level)
- **Adaptive Computation Time**: Dynamic computation steps based on problem difficulty
- **Transformer-based**: 4 layers, 256 hidden dimensions, 4 attention heads
- **Binary Output**: 2 output classes for queen placement prediction

### Training Configuration
- **Batch Size**: 512
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW
- **Checkpoint Frequency**: Every 250 steps
- **Curriculum Learning**: Progressive difficulty adjustment

## Data Format

Queens levels are stored as TypeScript files with the following structure:
```typescript
const level = {
  size: 8,
  colorRegions: [
    ["A", "A", "B", "B", "B", "C", "C", "C"],
    ["A", "D", "B", "D", "B", "E", "C", "C"],
    // ... more rows
  ],
  regionColors: {
    A: "color1",
    B: "color2",
    // ... color mappings
  }
}
```

## Key Innovations

1. **Binary Classification Approach**: Simplified from multi-class to binary prediction
2. **Real-World Data**: Using actual game levels instead of generated puzzles
3. **Geometric Augmentation**: 8x data multiplication through transformations
4. **Constraint-Aware Loss**: Handles Queens-specific rules in the loss function
5. **Flexible Grid Sizes**: Architecture supports various board sizes (focused on 8x8)

## Performance Metrics

The model tracks:
- **Output Loss**: Binary cross-entropy loss for queen placement
- **Accuracy**: Percentage of correctly predicted queen positions
- **Q-ACT Metrics**: Adaptive computation time halting accuracy
- **Curriculum Progress**: Automatic difficulty adjustment based on performance

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
