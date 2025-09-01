//
//  Linear.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import Foundation
import MLX
import MLXNN

public class Linear: Module, UnaryLayer {
    public let weight: MLXArray
    public let bias: MLXArray?

    public init(
        inDim: Int,
        outDim: Int,
        bias: Bool = true,
        dtype: DType = .float32,
        key: MLXArray? = nil
    ) {
        self.weight = truncNormalInit(
            [inDim, outDim], std: 1.0 / powf(Float(inDim), 0.5), dtype: dtype, key: key)
        self.bias = bias ? MLXArray.zeros([outDim], dtype: dtype) : nil
    }

    public init(weight: MLXArray, bias: MLXArray?) {
        self.weight = weight
        self.bias = bias
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x = matmul(x, weight)
        if let bias = bias {
            return x + bias
        } else {
            return x
        }
    }
}
