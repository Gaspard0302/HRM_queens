//
//  Attention.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import Foundation
import MLX
import MLXNN
import MLXRandom

public class Attention: Module {
    public let dim: Int
    public let headDim: Int
    public let numHeads: Int
    public let outputSize: Int
    public let keyValueHeadsPerHead: Int
    public let numKeyValueHeads: Int

    public let qkvProj: Linear
    public let outProj: Linear

    public init(
        dim: Int,
        headDim: Int,
        numHeads: Int,
        keyValueHeadsPerHead: Int,
        dtype: DType = .float32,
        key: MLXArray? = nil
    ) {
        self.dim = dim
        self.headDim = headDim
        self.numHeads = numHeads
        self.outputSize = headDim * numHeads
        self.keyValueHeadsPerHead = keyValueHeadsPerHead
        self.numKeyValueHeads = numHeads * keyValueHeadsPerHead

        let (qkvKey, outKey): (MLXArray?, MLXArray?) =
            if let key = key {
                MLXRandom.split(key: key)
            } else {
                (nil, nil)
            }
        self.qkvProj = Linear(
            inDim: dim,
            outDim: (numHeads + 2 * numKeyValueHeads) * headDim,
            bias: false,
            key: qkvKey
        )
        self.outProj = Linear(inDim: outputSize, outDim: dim, bias: false, key: outKey)
    }

    public func callAsFunction(_ x: MLXArray, rotaryPositionEmbedding: RotaryPositionEmbedding?)
        -> MLXArray
    {
        let (batchSize, seqLen, _) = x.shape3

        let qkv = qkvProj(x)
            .reshaped([batchSize, seqLen, numHeads + 2 * numKeyValueHeads, headDim])
        var query = qkv[0..., 0..., ..<numHeads]
        var key = qkv[0..., 0..., numHeads..<(numHeads + numKeyValueHeads)]
        var value = qkv[0..., 0..., (numHeads + numKeyValueHeads)...]

        if let rotaryPositionEmbedding {
            query = rotaryPositionEmbedding(query)
            key = rotaryPositionEmbedding(key)
        }

        query =
            query
            .reshaped([batchSize, seqLen, numKeyValueHeads, keyValueHeadsPerHead, headDim])
            .transposed(0, 2, 3, 1, 4)
        key =
            key
            .transposed(0, 2, 3, 1)
            .expandedDimensions(axis: 2)
        value =
            value
            .transposed(0, 2, 1, 3)
            .expandedDimensions(axis: 2)

        let attnLogits = matmul(query, key) * (1.0 / sqrtf(Float(headDim)))
        let attnWeights = softmax(attnLogits.asType(DType.float32), axis: -1).asType(
            attnLogits.dtype)
        let combined = matmul(attnWeights, value)
            .transposed(0, 3, 1, 2, 4)
            .reshaped([batchSize, seqLen, dim])

        return outProj(combined)
    }
}
