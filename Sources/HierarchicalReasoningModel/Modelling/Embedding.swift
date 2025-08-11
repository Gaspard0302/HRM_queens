//
//  Embedding.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import MLX
import MLXNN

public class Embedding: Module, UnaryLayer {
    public let embeddings: MLXArray

    public init(
        vocabSize: Int,
        dim: Int,
        initStd: Float,
        dtype: DType = .float32,
        key: MLXArray? = nil,
    ) {
        self.embeddings = truncNormalInit([vocabSize, dim], std: initStd, dtype: dtype, key: key)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return embeddings[x]
    }
}
