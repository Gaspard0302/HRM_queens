//
//  Training.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

public func sudokuLoss(
    model: HRMACTInner,
    hiddenStates: HRMACTInner.HiddenStates,
    boardInputs: MLXArray,
    boardTargets: MLXArray,
    segments: MLXArray,
    key: MLXArray? = nil
) -> [MLXArray] {
    let output = model(hiddenStates: hiddenStates, inputs: boardInputs)

    let outputLogits = output.output
    var outputLoss = crossEntropy(logits: outputLogits, targets: boardTargets, reduction: .none)
    let outputLossMask = (boardInputs .== 0).asType(outputLoss.dtype)
    outputLoss = outputLoss * outputLossMask

    let outputAccuracy = ((output.output.argMax(axis: 2) .== boardTargets) .|| (boardInputs .!= 0))
        .min(axis: 1)
    let qACTHaltTarget = outputAccuracy.asType(.int32)

    let nextSegments = segments + 1
    let isLastSegment = nextSegments .> model.config.act.haltMaxSteps
    var isHalted = isLastSegment .|| (output.qACTHalt .> output.qACTContinue)

    let (haltExplorationKey, minHaltSegmentsKey): (MLXArray?, MLXArray?) =
        if let key {
            MLXRandom.split(key: key)
        } else {
            (nil, nil)
        }
    let haltExploration =
        MLXRandom.uniform(low: 0.0, high: 1.0, output.qACTHalt.shape, key: haltExplorationKey)
        .< model.config.act.haltExplorationProbability
    let minHaltSegments =
        MLXRandom.randInt(
            2..<(model.config.act.haltMaxSteps + 1), segments.shape, key: minHaltSegmentsKey)
        * haltExploration.asType(.int32)
    isHalted = isHalted .&& (nextSegments .> minHaltSegments)

    let nextSegmentOutput = model(hiddenStates: output.hiddenStates, inputs: boardInputs)
    let nextQACTHalt = stopGradient(nextSegmentOutput.qACTHalt)
    let nextQACTContinue = stopGradient(nextSegmentOutput.qACTContinue)

    let qACTContinueTarget = sigmoid(
        which(
            isLastSegment,
            nextQACTHalt,
            maximum(nextQACTHalt, nextQACTContinue)
        ))

    let qACTLoss =
        (binaryCrossEntropy(logits: output.qACTHalt, targets: qACTHaltTarget, reduction: .none)
            + binaryCrossEntropy(
                logits: output.qACTContinue, targets: qACTContinueTarget, reduction: .none)) / 2

    let avgOutputLoss = outputLoss.sum() / outputLossMask.sum()
    let avgQACTLoss = qACTLoss.mean()

    let avgOutputFullAccuracy =
        ((output.output.argMax(axis: 2) .== boardTargets) .|| (boardInputs .!= 0)).mean()
    let avgQACTHaltAccuracy = ((output.qACTHalt .>= 0) .== outputAccuracy).mean()

    return [
        avgOutputLoss + avgQACTLoss,
        avgOutputLoss,
        avgQACTLoss,
        isHalted,
        avgOutputFullAccuracy,
        avgQACTHaltAccuracy,
        output.hiddenStates.highLevel,
        output.hiddenStates.lowLevel,
    ]
}

public struct TrainingBatch: Evaluatable {
    private static let DIFFICULTIES: [Difficulty] = [
        .easy, .medium, .hard, .extreme,
    ]
    private static let CURRICULUM_DIFFICULTY_PROBAS: [[Float]] = [
        [1.0, 0.0, 0.0, 0.0],  // stage 0: only easy
        [0.7, 0.3, 0.0, 0.0],  // stage 1: mostly easy, some medium
        [0.5, 0.4, 0.1, 0.0],  // stage 2: mix of easy, medium, some hard
        [0.3, 0.3, 0.3, 0.1],  // stage 3: mix of all difficulties
        [0.1, 0.3, 0.4, 0.2],  // stage 4: more hard and extreme
    ]

    private let initialHiddenStates: HRMACTInner.HiddenStates

    public var hiddenStates: HRMACTInner.HiddenStates
    public var boardInputs: MLXArray
    public var boardTargets: MLXArray
    public var segments: MLXArray

    public var curriculumLevel: Int = 0
    public var totalPuzzles: Int = 0

    public init(initialHiddenState: HRMACTInner.HiddenStates, size: Int) {
        self.initialHiddenStates = initialHiddenState
        self.hiddenStates = .init(
            highLevel: MLXArray.zeros(
                [size, 1] + initialHiddenStates.highLevel.shape,
                dtype: initialHiddenStates.highLevel.dtype),
            lowLevel: MLXArray.zeros(
                [size, 1] + initialHiddenStates.lowLevel.shape,
                dtype: initialHiddenStates.lowLevel.dtype)
        )
        self.boardInputs = MLXArray.zeros([size, 81], dtype: .int32)
        self.boardTargets = MLXArray.zeros([size, 81], dtype: .int32)
        self.segments = MLXArray.zeros([size], dtype: .int32)

        for i in 0..<size {
            self.replace(sampleAt: i)
        }
    }

    public func innerState() -> [MLXArray] {
        hiddenStates.innerState() + [boardInputs, boardTargets, segments]
    }

    private func sampleDifficulty() -> Difficulty {
        let probabilities = Self.CURRICULUM_DIFFICULTY_PROBAS[curriculumLevel]
        let rand = Float.random(in: 0..<1)
        var cumulativeProbability: Float = 0.0
        for (index, probability) in probabilities.enumerated() {
            cumulativeProbability += probability
            if rand < cumulativeProbability {
                return Self.DIFFICULTIES[index]
            }
        }
        fatalError()
    }

    public mutating func replace(sampleAt idx: Int) {
        hiddenStates.highLevel[idx] = initialHiddenStates.highLevel[.newAxis]
        hiddenStates.lowLevel[idx] = initialHiddenStates.lowLevel[.newAxis]
        segments[idx] = MLXArray(0, dtype: .int32)

        let (puzzle, solution) = generateSudoku(difficulty: sampleDifficulty())
        boardInputs[idx] = MLXArray(puzzle.flatMap({ $0 })).asType(.int32)
        boardTargets[idx] = MLXArray(solution.flatMap({ $0 })).asType(.int32)

        self.totalPuzzles += 1
    }

    public mutating func graduate() {
        let nextCurriculumLevel = curriculumLevel + 1
        guard nextCurriculumLevel < Self.CURRICULUM_DIFFICULTY_PROBAS.count else {
            print("Reached highest curriculum level, cannot graduate.")
            return
        }

        curriculumLevel = nextCurriculumLevel
        print("Graduated to level \(curriculumLevel).")
    }
}

public func step(
    model: HRMACTInner,
    optimizer: Optimizer,
    batch: inout TrainingBatch,
    key: MLXArray
) -> (output: (Float, Float), qACT: (Float, Float)) {
    let (data, grad) = valueAndGrad(model: model) {
        (model: HRMACTInner, args: [MLXArray]) -> [MLXArray] in
        let (hlZ, llZ, boardX, boardY, seg, key) = (
            args[0], args[1], args[2], args[3], args[4], args[5]
        )

        return sudokuLoss(
            model: model,
            hiddenStates: .init(highLevel: hlZ, lowLevel: llZ),
            boardInputs: boardX,
            boardTargets: boardY,
            segments: seg,
            key: key
        )
    }(
        model,
        [
            batch.hiddenStates.highLevel,
            batch.hiddenStates.lowLevel,
            batch.boardInputs,
            batch.boardTargets,
            batch.segments,
            key,
        ])

    optimizer.update(model: model, gradients: grad)

    let (outputLoss, qACTLoss, outputAcc, qACTAcc): (Float, Float, Float, Float) = (
        data[1].item(), data[2].item(), data[4].item(), data[5].item()
    )
    print(
        "Output [\(outputLoss) \(outputAcc)] | Q-ACT [\(qACTLoss) \(qACTAcc)] | Puzzles [\(batch.totalPuzzles)] | Curriculum Level [\(batch.curriculumLevel)]"
    )

    let (nextHLState, nextLLState) = (data[6], data[7])
    batch.hiddenStates.highLevel = nextHLState
    batch.hiddenStates.lowLevel = nextLLState
    batch.segments += 1

    let isHalted = data[3].asArray(Bool.self)
    for (idx, isHalted) in isHalted.enumerated() {
        if isHalted {
            batch.replace(sampleAt: idx)
        }
    }

    return ((outputLoss, outputAcc), (qACTLoss, qACTAcc))
}
