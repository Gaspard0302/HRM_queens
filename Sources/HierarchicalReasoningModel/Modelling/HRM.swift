//
//  HRM.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import Foundation
import MLX
import MLXNN
import MLXRandom

public struct HRMACTModelConfig {
    public struct TransformerConfig {
        public let numLayers: Int
        public let hiddenSize: Int
        public let numHeads: Int
        public let expansion: Float
        public let normEpsilon: Float = 1e-5
        public let ropeTheta: Float = 10000.0
    }

    public struct ACTConfig {
        public let haltMaxSteps: Int
        public let haltExplorationProbability: Float
    }

    public let seqLen: Int
    public let vocabSize: Int

    public let highLevelCycles: Int
    public let lowLevelCycles: Int

    public let transformers: TransformerConfig

    public let act: ACTConfig

    public let dtype: DType = .bfloat16
}

public class HRMACTBlock: Module {
    public let selfAttn: Attention
    public let mlp: SwiGLU
    public let normEpsilon: Float

    public init(
        hiddenSize: Int,
        numHeads: Int,
        expansion: Float,
        normEpsilon: Float,
        dtype: DType = .float32,
        key: MLXArray? = nil
    ) {
        let (selfAttnKey, mlpKey): (MLXArray?, MLXArray?) =
            if let key = key {
                MLXRandom.split(key: key)
            } else {
                (nil, nil)
            }
        self.selfAttn = Attention(
            dim: hiddenSize,
            headDim: hiddenSize / numHeads,
            numHeads: numHeads,
            keyValueHeadsPerHead: 1,
            dtype: dtype,
            key: selfAttnKey
        )
        self.mlp = SwiGLU(dim: hiddenSize, expansion: expansion, dtype: dtype, key: mlpKey)
        self.normEpsilon = normEpsilon
    }

    public func callAsFunction(_ x: MLXArray, rotaryPositionEmbedding: RotaryPositionEmbedding?)
        -> MLXArray
    {
        var x = x
        x = rmsNorm(
            x + selfAttn(x, rotaryPositionEmbedding: rotaryPositionEmbedding), epsilon: normEpsilon)
        x = rmsNorm(x + mlp(x), epsilon: normEpsilon)
        return x
    }
}

public class HRMACTReasoner: Module {
    public let blocks: [HRMACTBlock]

    public init(
        numLayers: Int,
        hiddenSize: Int,
        numHeads: Int,
        expansion: Float,
        normEpsilon: Float,
        dtype: DType = .float32,
        key: MLXArray? = nil
    ) {
        let keys =
            if let key = key {
                MLXRandom.split(key: key, into: numLayers)
            } else {
                [MLXArray?](repeating: nil, count: numLayers)
            }
        self.blocks = keys.map { key in
            HRMACTBlock(
                hiddenSize: hiddenSize,
                numHeads: numHeads,
                expansion: expansion,
                normEpsilon: normEpsilon,
                dtype: dtype,
                key: key
            )
        }
    }

    public func callAsFunction(
        hiddenState: MLXArray, inputInjection: MLXArray,
        rotaryPositionEmbedding: RotaryPositionEmbedding?
    ) -> MLXArray {
        var hiddenState = hiddenState + inputInjection
        for block in blocks {
            hiddenState = block(hiddenState, rotaryPositionEmbedding: rotaryPositionEmbedding)
        }
        return hiddenState
    }
}

public class HRMACTInner: Module {
    private class InitialHiddenStates: Module {
        let highLevel: MLXArray
        let lowLevel: MLXArray

        init(hiddenSize: Int, dtype: DType = .float32, key: MLXArray? = nil) {
            let (hlKey, llKey): (MLXArray?, MLXArray?) =
                if let key = key {
                    MLXRandom.split(key: key)
                } else {
                    (nil, nil)
                }
            self.highLevel = truncNormalInit([hiddenSize], std: 1.0, dtype: dtype, key: hlKey)
            self.lowLevel = truncNormalInit([hiddenSize], std: 1.0, dtype: dtype, key: llKey)

            super.init()
            self.freeze()
        }
    }

    public struct HiddenStates: Evaluatable {
        public var highLevel: MLXArray
        public var lowLevel: MLXArray

        public func innerState() -> [MLXArray] {
            [highLevel, lowLevel]
        }

        public func map(_ transform: (MLXArray) -> (MLXArray)) -> HiddenStates {
            HiddenStates(
                highLevel: transform(highLevel),
                lowLevel: transform(lowLevel)
            )
        }
    }

    public struct Output: Evaluatable {
        public let hiddenStates: HiddenStates
        public let output: MLXArray
        public let qACTHalt: MLXArray
        public let qACTContinue: MLXArray

        public func innerState() -> [MLXArray] {
            hiddenStates.innerState() + [output, qACTHalt, qACTContinue]
        }
    }

    public let config: HRMACTModelConfig

    public let clsToken: MLXArray
    public let inputEmbedding: Embedding
    public let outputHead: Linear
    public let qACTHead: Linear

    public let rotaryEmb: RotaryPositionEmbedding
    public let highLevelReasoner: HRMACTReasoner
    public let lowLevelReasoner: HRMACTReasoner

    // TODO: Is this safe — will the parameters still be saved and loaded correctly?
    private let initialHiddenStatesParams: InitialHiddenStates
    public var initialHiddenStates: HiddenStates {
        HiddenStates(
            highLevel: initialHiddenStatesParams.highLevel,
            lowLevel: initialHiddenStatesParams.lowLevel
        )
    }

    public init(config: HRMACTModelConfig, key: MLXArray) {
        self.config = config

        var key = key

        let clsTokenKey: MLXArray
        (key, clsTokenKey) = MLXRandom.split(key: key)
        self.clsToken = truncNormalInit(
            [config.transformers.hiddenSize],
            std: 1.0 / sqrtf(Float(config.transformers.hiddenSize)), dtype: config.dtype,
            key: clsTokenKey)

        let embeddingKey: MLXArray
        (key, embeddingKey) = MLXRandom.split(key: key)
        self.inputEmbedding = Embedding(
            vocabSize: config.vocabSize,
            dim: config.transformers.hiddenSize,
            initStd: 1.0 / sqrtf(Float(config.transformers.hiddenSize)),
            dtype: config.dtype,
            key: embeddingKey
        )

        let outputKey: MLXArray
        (key, outputKey) = MLXRandom.split(key: key)
        self.outputHead = Linear(
            inDim: config.transformers.hiddenSize,
            outDim: config.vocabSize,
            bias: false,
            dtype: config.dtype,
            key: outputKey
        )

        self.qACTHead = Linear(
            weight: MLXArray.zeros([config.transformers.hiddenSize, 2], dtype: config.dtype),
            bias: MLXArray.zeros([2], dtype: config.dtype) - 5
        )

        self.rotaryEmb = RotaryPositionEmbedding(
            dim: config.transformers.hiddenSize / config.transformers.numHeads,
            maxLength: config.seqLen + 1,  // + CLS token
            base: config.transformers.ropeTheta,
            dtype: config.dtype
        )

        let highLevelReasonerKey: MLXArray
        (key, highLevelReasonerKey) = MLXRandom.split(key: key)
        self.highLevelReasoner = HRMACTReasoner(
            numLayers: config.transformers.numLayers,
            hiddenSize: config.transformers.hiddenSize,
            numHeads: config.transformers.numHeads,
            expansion: config.transformers.expansion,
            normEpsilon: config.transformers.normEpsilon,
            dtype: config.dtype,
            key: highLevelReasonerKey
        )

        let lowLevelReasonerKey: MLXArray
        (key, lowLevelReasonerKey) = MLXRandom.split(key: key)
        self.lowLevelReasoner = HRMACTReasoner(
            numLayers: config.transformers.numLayers,
            hiddenSize: config.transformers.hiddenSize,
            numHeads: config.transformers.numHeads,
            expansion: config.transformers.expansion,
            normEpsilon: config.transformers.normEpsilon,
            dtype: config.dtype,
            key: lowLevelReasonerKey
        )

        self.initialHiddenStatesParams = InitialHiddenStates(
            hiddenSize: config.transformers.hiddenSize,
            dtype: config.dtype,
            key: key
        )
    }

    public func callAsFunction(hiddenStates: HiddenStates, inputs: MLXArray) -> Output {
        let inputEmbeddings =
            concatenated(
                [
                    stacked([MLXArray](repeating: clsToken[.newAxis], count: inputs.shape[0])),
                    inputEmbedding(inputs),
                ],
                axis: 1
            ) * sqrtf(Float(config.transformers.hiddenSize))

        var (lowLevelZ, highLevelZ) = (hiddenStates.lowLevel, hiddenStates.highLevel)
        for cycle in 1...(config.highLevelCycles * config.lowLevelCycles - 1) {
            lowLevelZ = lowLevelReasoner(
                hiddenState: lowLevelZ,
                inputInjection: highLevelZ + inputEmbeddings,
                rotaryPositionEmbedding: rotaryEmb
            )
            eval(lowLevelZ)
            if cycle % config.lowLevelCycles == 0 {
                highLevelZ = highLevelReasoner(
                    hiddenState: highLevelZ,
                    inputInjection: lowLevelZ,
                    rotaryPositionEmbedding: rotaryEmb
                )
                eval(highLevelZ)
            }
        }

        lowLevelZ = stopGradient(lowLevelZ)
        highLevelZ = stopGradient(highLevelZ)

        lowLevelZ = lowLevelReasoner(
            hiddenState: lowLevelZ,
            inputInjection: highLevelZ + inputEmbeddings,
            rotaryPositionEmbedding: rotaryEmb
        )
        highLevelZ = highLevelReasoner(
            hiddenState: highLevelZ,
            inputInjection: lowLevelZ,
            rotaryPositionEmbedding: rotaryEmb
        )

        let outputLogits = outputHead(highLevelZ[0..., 1...])
        let qACTLogits = qACTHead(highLevelZ[0..., 0])

        return Output(
            hiddenStates: .init(
                highLevel: stopGradient(highLevelZ), lowLevel: stopGradient(lowLevelZ)),
            output: outputLogits,
            qACTHalt: qACTLogits[0..., 0],
            qACTContinue: qACTLogits[0..., 1]
        )
    }
}
