// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "HierarchicalReasoningModel",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "HierarchicalReasoningModel", targets: ["HierarchicalReasoningModel"]),
    ],
    dependencies: [
        .package(name: "mlx-swift", path: "/Users/tanmaybakshi/mlx-swift")
    ],
    targets: [
        .executableTarget(
            name: "HierarchicalReasoningModel",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
    ]
)
