//
//  SwiGLU.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import MLX
import MLXNN
import MLXRandom

public class SwiGLU: Module, UnaryLayer {
    public let gateUpProj: Linear
    public let downProj: Linear

    public init(
        dim: Int,
        expansion: Float,
        dtype: DType = .float32,
        key: MLXArray? = nil
    ) {
        let (upKey, downKey): (MLXArray?, MLXArray?) =
            if let key = key {
                MLXRandom.split(key: key)
            } else {
                (nil, nil)
            }
        let interDim = Self.findMultiple(Int(expansion * Float(dim) * 2.0 / 3.0), 256)
        self.gateUpProj = Linear(inDim: dim, outDim: interDim * 2, bias: false, key: upKey)
        self.downProj = Linear(inDim: interDim, outDim: dim, bias: false, key: downKey)
    }

    private static func findMultiple(_ a: Int, _ b: Int) -> Int {
        (-(a / -b)) * b
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (gate, up) = gateUpProj(x).split(axis: -1)
        return downProj(silu(gate) * up)
    }
}
