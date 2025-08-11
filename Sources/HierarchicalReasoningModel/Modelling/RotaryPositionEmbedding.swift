//
//  RotaryPositionEmbedding.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import MLX
import MLXNN

public class RotaryPositionEmbedding: Module {
    public let cos: MLXArray
    public let sin: MLXArray

    public init(
        dim: Int,
        maxLength: Int,
        base: Float,
        dtype: DType = .float32
    ) {
        let invFreq =
            1.0
            / MLXArray(base, dtype: dtype).pow(
                MLXArray(stride(from: 0, through: dim - 1, by: 2)).asType(dtype) / Float(dim))
        let t = MLXArray(0..<maxLength).asType(dtype)
        let freqs = outer(t, invFreq)

        let emb = concatenated([freqs, freqs], axis: -1)
            .expandedDimensions(axis: -2)
        self.cos = emb.cos()
        self.sin = emb.sin()

        super.init()
        self.freeze()
    }

    private static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let x = x.movedAxis(source: -1, destination: 0)
        let halfDim = x.shape[0] / 2
        let x1 = x[..<halfDim]
        let x2 = x[halfDim...]
        return concatenated([-x2, x1], axis: 0).movedAxis(source: 0, destination: -1)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        (x * self.cos) + (Self.rotateHalf(x) * self.sin)
    }
}
