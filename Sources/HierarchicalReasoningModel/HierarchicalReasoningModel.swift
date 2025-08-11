//
//  HierarchicalReasoningModel.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

@main
struct HierarchicalReasoningModel {
    static func train() throws {
        var key = MLXRandom.key(42)

        let modelKey: MLXArray
        (key, modelKey) = MLXRandom.split(key: key)
        let model = HRMACTInner(
            config: HRMACTModelConfig(
                seqLen: 9 * 9,
                vocabSize: 10,
                highLevelCycles: 2,
                lowLevelCycles: 2,
                transformers: .init(numLayers: 4, hiddenSize: 256, numHeads: 4, expansion: 4),
                act: .init(haltMaxSteps: 16, haltExplorationProbability: 0.1)
            ),
            key: modelKey
        )
        eval(model)

        let optimizer = AdamAtan2(learningRate: 1e-4, betas: (0.9, 0.95))

        var batch = TrainingBatch(initialHiddenState: model.initialHiddenStates, size: 512)

        var stepIdx = 0
        var stepsSinceGraduation = 0
        var accuracyHistory: [Float] = [Float](repeating: 0, count: 300)
        while true {
            stepIdx += 1
            stepsSinceGraduation += 1
            print("Step \(stepIdx)")

            let stepKey: MLXArray
            (key, stepKey) = MLXRandom.split(key: key)
            let ((_, outputAcc), (_, _)) = step(
                model: model, optimizer: optimizer, batch: &batch, key: stepKey)

            if stepIdx == 1 || stepIdx % 250 == 0 {
                try MLX.save(
                    arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
                    url: URL(filePath: "checkpoint-\(stepIdx).safetensors")
                )
            }

            accuracyHistory.removeFirst()
            accuracyHistory.append(outputAcc)
            let avgRollingAccuracy = accuracyHistory.reduce(0, +) / Float(accuracyHistory.count)
            if avgRollingAccuracy >= 0.85 && stepsSinceGraduation >= 300 {
                stepsSinceGraduation = 0
                batch.graduate()
            }

            print()
        }
    }

    static func infer(checkpoint: String, difficulty: Difficulty) throws {
        var key = MLXRandom.key(42)

        let weights = try MLX.loadArrays(url: URL(filePath: checkpoint))
        let parameters = ModuleParameters.unflattened(weights)

        let modelKey: MLXArray
        (key, modelKey) = MLXRandom.split(key: key)
        let model = try HRMACTInner(
            config: HRMACTModelConfig(
                seqLen: 9 * 9,
                vocabSize: 10,
                highLevelCycles: 2,
                lowLevelCycles: 2,
                transformers: .init(numLayers: 4, hiddenSize: 256, numHeads: 4, expansion: 4),
                act: .init(haltMaxSteps: 16, haltExplorationProbability: 0.1)
            ),
            key: modelKey
        ).update(parameters: parameters, verify: .all)
        eval(model)
        print("Loaded model!")

        let (rawPuzzle, rawSolution) = generateSudoku(difficulty: difficulty)
        print("Puzzle:\n\(sudokuBoardString(rawPuzzle))")
        print("Solution:\n\(sudokuBoardString(rawSolution))")

        let puzzleIn = MLXArray(rawPuzzle.flatMap { $0 }).asType(.int32)[.newAxis]
        var hiddenStates = model
            .initialHiddenStates
            .map { $0[.newAxis, .newAxis] }
        for segment in 1...model.config.act.haltMaxSteps {
            print("\nSegment \(segment)")

            let output = model(hiddenStates: hiddenStates, inputs: puzzleIn)
            hiddenStates = output.hiddenStates

            let predictions = output.output[0].argMax(axis: 1)

            var accurateSquares = 0
            var predictedSquares = 0
            let predictedFlatBoard = zip(
                zip(rawPuzzle.flatMap({ $0 }), rawSolution.flatMap({ $0 })),
                predictions.asArray(Int.self)
            )
            .map { puzz, predictedSquare in
                let (puzzleSquare, solutionSquare) = puzz

                if puzzleSquare != 0 {
                    return puzzleSquare
                }

                accurateSquares += predictedSquare == solutionSquare ? 1 : 0
                predictedSquares += 1

                return predictedSquare
            }
            let predictedBoard = stride(from: 0, through: 9 * 9 - 1, by: 9)
                .map { Array(predictedFlatBoard[$0..<($0 + 9)]) }
            print(
                "Predicted solution (\(accurateSquares) / \(predictedSquares)):\n\(sudokuBoardString(predictedBoard))"
            )

            let (qHalt, qContinue): (Float, Float) = (
                sigmoid(output.qACTHalt[0]).item(), sigmoid(output.qACTContinue[0]).item()
            )
            print("Q (halt - continue): \(qHalt) - \(qContinue)")

            if qHalt > qContinue {
                print("Halting.")
                break
            }
        }
    }

    static func main() throws {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            print("Usage: \(args[0]) train | infer <checkpoint-path> <difficulty>")
            return
        }
        let mode = args[1]

        switch mode {
        case "train":
            try train()
        case "infer":
            guard args.count >= 4 else {
                print("Usage: \(args[0]) infer <checkpoint-path> <difficulty>")
                return
            }
            let checkpoint = args[2]
            let diffStr = args[3]

            let difficulty: Difficulty
            switch diffStr {
            case "very-easy": difficulty = .veryEasy
            case "easy": difficulty = .easy
            case "medium": difficulty = .medium
            case "hard": difficulty = .hard
            case "extreme": difficulty = .extreme
            default:
                fatalError(
                    "Unknown difficulty: \(diffStr). Expected one of veryEasy, easy, medium, hard, extreme."
                )
            }

            try infer(checkpoint: checkpoint, difficulty: difficulty)
        default:
            fatalError("Unknown mode: \(mode). Expected 'train' or 'infer'.")
        }
    }
}
