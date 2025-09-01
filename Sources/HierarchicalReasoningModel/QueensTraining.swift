//
//  QueensTraining.swift
//  HierarchicalReasoningModel
//
//  Training system adapted for Queens game
//

import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import Foundation

public func queensLoss(
    model: HRMACTInner,
    hiddenStates: HRMACTInner.HiddenStates,
    boardInputs: MLXArray,
    boardTargets: MLXArray,
    regionInputs: MLXArray,  // Region constraints
    segments: MLXArray,
    key: MLXArray? = nil
) -> [MLXArray] {
    let output = model(hiddenStates: hiddenStates, inputs: boardInputs)

    let outputLogits = output.output
    
    // For Queens, we treat this as binary classification per cell
    // We use the first output class (0) as "no queen" and second class (1) as "queen"
    // Convert to logits for binary classification
    // outputLogits has shape [batch_size, seq_len, vocab_size] where vocab_size=2
    let queenLogits = outputLogits[0..., 0..., 1] - outputLogits[0..., 0..., 0] // logit difference gives us binary logit
    
    let targetFloats = boardTargets.asType(outputLogits.dtype)
    
    // For Queens, we want to predict all cells (not just empty ones)
    // But we can still mask if needed for partial puzzles
    let allMask = MLXArray.ones(boardInputs.shape, dtype: outputLogits.dtype)
    var outputLoss = binaryCrossEntropy(logits: queenLogits, targets: targetFloats, reduction: .none)
    outputLoss = outputLoss * allMask

    // For accuracy, check if we correctly predict queen placements
    let binaryPredictions = (sigmoid(queenLogits) .> 0.5).asType(.int32)
    let outputAccuracy = (binaryPredictions .== boardTargets)
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

    let avgOutputLoss = outputLoss.sum() / allMask.sum()
    let avgQACTLoss = qACTLoss.mean()

    let avgOutputFullAccuracy = outputAccuracy.mean()
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

public struct QueensTrainingBatch: Evaluatable {
    private static let STANDARD_SIZE = 8  // Use 8x8 as our standard size
    
    private let initialHiddenStates: HRMACTInner.HiddenStates
    private var queensLevels: [QueensLevel] = []
    private var augmentedLevels: [QueensLevel] = []

    public var hiddenStates: HRMACTInner.HiddenStates
    public var boardInputs: MLXArray     // 0 = empty, 1 = given queen (for partial puzzles)
    public var boardTargets: MLXArray    // 0 = no queen, 1 = place queen
    public var regionInputs: MLXArray    // Encoded region information
    public var segments: MLXArray

    public var curriculumLevel: Int = 0
    public var totalPuzzles: Int = 0

    public init(initialHiddenState: HRMACTInner.HiddenStates, size: Int, queensLevelsDirectory: String) {
        self.initialHiddenStates = initialHiddenState
        self.hiddenStates = .init(
            highLevel: MLXArray.zeros(
                [size, 1] + initialHiddenStates.highLevel.shape,
                dtype: initialHiddenStates.highLevel.dtype),
            lowLevel: MLXArray.zeros(
                [size, 1] + initialHiddenStates.lowLevel.shape,
                dtype: initialHiddenStates.lowLevel.dtype)
        )
        
        let seqLen = Self.STANDARD_SIZE * Self.STANDARD_SIZE
        self.boardInputs = MLXArray.zeros([size, seqLen], dtype: .int32)
        self.boardTargets = MLXArray.zeros([size, seqLen], dtype: .int32)
        self.regionInputs = MLXArray.zeros([size, seqLen], dtype: .int32)
        self.segments = MLXArray.zeros([size], dtype: .int32)

        // Load Queens levels
        print("Loading Queens levels from: \(queensLevelsDirectory)")
        let allLevels = loadQueensLevels(from: queensLevelsDirectory)
        
        // Filter to standard size and create augmentations
        let standardLevels = allLevels.filter { $0.size == Self.STANDARD_SIZE }
        print("Found \(standardLevels.count) levels of size \(Self.STANDARD_SIZE)")
        
        self.queensLevels = standardLevels
        
        // Generate augmented data (8x more data)
        for level in standardLevels {
            let augmentations = generateAugmentations(from: level)
            augmentedLevels.append(contentsOf: augmentations)
        }
        
        print("Created \(augmentedLevels.count) total training examples (including augmentations)")

        for i in 0..<size {
            self.replace(sampleAt: i)
        }
    }

    public func innerState() -> [MLXArray] {
        hiddenStates.innerState() + [boardInputs, boardTargets, regionInputs, segments]
    }

    public mutating func replace(sampleAt idx: Int) {
        hiddenStates.highLevel[idx] = initialHiddenStates.highLevel[.newAxis]
        hiddenStates.lowLevel[idx] = initialHiddenStates.lowLevel[.newAxis]
        segments[idx] = MLXArray(0, dtype: .int32)

        guard !augmentedLevels.isEmpty else {
            print("No Queens levels loaded!")
            return
        }

        // Randomly select a level
        let levelIndex = Int.random(in: 0..<augmentedLevels.count)
        let level = augmentedLevels[levelIndex]
        
        // Generate solution for this level
        guard let solution = solveQueens(level: level) else {
            print("Could not solve level \(levelIndex), trying another...")
            replace(sampleAt: idx) // Retry with different level
            return
        }
        
        // Convert level to input format
        let regions = level.regionToNumeric()
        let flatRegions = regions.flatMap { $0 }
        let flatSolution = solution.flatMap { $0 }
        
        // For training, we start with empty board and predict full solution
        // In real scenarios, some queens might be given as clues
        let emptyBoard = Array(repeating: 0, count: Self.STANDARD_SIZE * Self.STANDARD_SIZE)
        
        boardInputs[idx] = MLXArray(emptyBoard).asType(.int32)
        boardTargets[idx] = MLXArray(flatSolution).asType(.int32)
        regionInputs[idx] = MLXArray(flatRegions).asType(.int32)

        self.totalPuzzles += 1
    }

    public mutating func graduate() {
        let nextCurriculumLevel = curriculumLevel + 1
        // For Queens, we can keep it simple with just one curriculum level
        // or implement progressive difficulty based on region complexity
        curriculumLevel = nextCurriculumLevel
        print("Graduated to level \(curriculumLevel).")
    }
}

public func queensStep(
    model: HRMACTInner,
    optimizer: Optimizer,
    batch: inout QueensTrainingBatch,
    key: MLXArray
) -> (output: (Float, Float), qACT: (Float, Float)) {
    let (data, grad) = valueAndGrad(model: model) {
        (model: HRMACTInner, args: [MLXArray]) -> [MLXArray] in
        let (hlZ, llZ, boardX, boardY, regionX, seg, key) = (
            args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        )

        return queensLoss(
            model: model,
            hiddenStates: .init(highLevel: hlZ, lowLevel: llZ),
            boardInputs: boardX,
            boardTargets: boardY,
            regionInputs: regionX,
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
            batch.regionInputs,
            batch.segments,
            key,
        ])

    optimizer.update(model: model, gradients: grad)

    let (outputLoss, qACTLoss, outputAcc, qACTAcc): (Float, Float, Float, Float) = (
        data[1].item(), data[2].item(), data[4].item(), data[5].item()
    )
    print(
        "Queens Output [\(outputLoss) \(outputAcc)] | Q-ACT [\(qACTLoss) \(qACTAcc)] | Puzzles [\(batch.totalPuzzles)] | Curriculum Level [\(batch.curriculumLevel)]"
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
