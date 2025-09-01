//
//  TruncNormalInit.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import Foundation
import MLX
import MLXRandom

public func truncNormalInit(
    _ shape: [Int],
    std: Float = 1.0,
    lower: Float = -2.0,
    upper: Float = 2.0,
    dtype: DType = .float32,
    key: MLXArray? = nil
) -> MLXArray {
    if std == 0.0 {
        return MLXArray.zeros(shape, dtype: dtype)
    }

    let sqrt2 = sqrtf(2.0)
    let a = erff(lower / sqrt2)
    let b = erff(upper / sqrt2)
    let z = (b - a) / 2

    let c = powf(2 * Float.pi, -0.5)
    let pdfU = c * expf(-0.5 * powf(lower, 2))
    let pdfL = c * expf(-0.5 * powf(upper, 2))
    let compStd =
        std / sqrtf(1.0 - (upper * pdfU - lower * pdfL) / z - powf(((pdfU - pdfL) / z), 2))

    var result = MLXRandom.uniform(low: a, high: b, shape, key: key)
    result = erfInverse(result)
    result = result * (sqrt2 * compStd)
    result = clip(result, min: lower * compStd, max: upper * compStd)

    return result
}
