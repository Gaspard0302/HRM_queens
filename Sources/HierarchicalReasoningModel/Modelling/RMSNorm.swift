//
//  RMSNorm.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import MLX

public func rmsNorm(_ x: MLXArray, epsilon: Float = 1e-6) -> MLXArray {
    let originalDtype = x.dtype
    let x = x.asType(.float32)

    let variance = x.square().mean(axis: -1, keepDims: true)
    return (x * rsqrt(variance + epsilon)).asType(originalDtype)
}
