//
//  Queens.swift
//  HierarchicalReasoningModel
//
//  Queens game logic and data structures
//

import Foundation

// MARK: - Data Structures

public struct QueensLevel {
    public let size: Int
    public let colorRegions: [[String]]
    public let regionColors: [String: String]? // Optional, mainly for visualization
    
    public init(size: Int, colorRegions: [[String]], regionColors: [String: String]? = nil) {
        self.size = size
        self.colorRegions = colorRegions
        self.regionColors = regionColors
    }
    
    /// Get unique region identifiers
    public var regions: Set<String> {
        return Set(colorRegions.flatMap { $0 })
    }
    
    /// Convert region letters to numeric IDs (0-based)
    public func regionToNumeric() -> [[Int]] {
        let uniqueRegions = Array(regions).sorted()
        let regionMap = Dictionary(uniqueKeysWithValues: uniqueRegions.enumerated().map { ($1, $0) })
        
        return colorRegions.map { row in
            row.map { regionMap[$0] ?? 0 }
        }
    }
}

public enum QueensDifficulty {
    case easy
    case medium 
    case hard
    case extreme
}

// MARK: - Queens Game Logic

/// Check if it's safe to place a queen at the given position
public func isSafeToPlaceQueen(
    board: [[Int]], // 0 = empty, 1 = queen
    row: Int,
    col: Int,
    size: Int,
    colorRegions: [[Int]] // Numeric region IDs
) -> Bool {
    let region = colorRegions[row][col]
    
    // Check for same row and column
    for i in 0..<size {
        if board[row][i] == 1 || board[i][col] == 1 {
            return false
        }
    }
    
    // Check adjacent diagonal squares
    let adjacentDiagonals = [
        (row - 1, col - 1),
        (row - 1, col + 1),
        (row + 1, col - 1),
        (row + 1, col + 1),
    ]
    
    for (r, c) in adjacentDiagonals {
        if r >= 0 && r < size && c >= 0 && c < size && board[r][c] == 1 {
            return false
        }
    }
    
    // Check for the same region
    for r in 0..<size {
        for c in 0..<size {
            if colorRegions[r][c] == region && board[r][c] == 1 {
                return false
            }
        }
    }
    
    return true
}

/// Check if the board satisfies all win conditions
public func checkWinCondition(
    board: [[Int]], // 0 = empty, 1 = queen
    size: Int,
    colorRegions: [[Int]]
) -> Bool {
    var queensPerRow = Array(repeating: 0, count: size)
    var queensPerCol = Array(repeating: 0, count: size)
    var queensPerRegion: [Int: Int] = [:]
    
    // Track diagonal positions for adjacency checking
    var mainDiagonal: [Int: [Int]] = [:]
    var antiDiagonal: [Int: [Int]] = [:]
    
    for row in 0..<size {
        for col in 0..<size {
            if board[row][col] == 1 {
                // Count queens per row and column
                queensPerRow[row] += 1
                queensPerCol[col] += 1
                
                // Count queens per region
                let region = colorRegions[row][col]
                queensPerRegion[region, default: 0] += 1
                
                // Track diagonal positions
                let mainDiagIndex = row - col
                mainDiagonal[mainDiagIndex, default: []].append(row)
                
                let antiDiagIndex = row + col
                antiDiagonal[antiDiagIndex, default: []].append(row)
            }
        }
    }
    
    // Check exactly 1 queen per row and column
    for i in 0..<size {
        if queensPerRow[i] != 1 || queensPerCol[i] != 1 {
            return false
        }
    }
    
    // Check exactly 1 queen per region
    for (_, count) in queensPerRegion {
        if count != 1 {
            return false
        }
    }
    
    // Check for diagonal adjacency violations
    for (_, rows) in mainDiagonal {
        if hasAdjacent(rows: rows) {
            return false
        }
    }
    
    for (_, rows) in antiDiagonal {
        if hasAdjacent(rows: rows) {
            return false
        }
    }
    
    return true
}

/// Helper function to check if queens are placed adjacently in a diagonal
private func hasAdjacent(rows: [Int]) -> Bool {
    let sortedRows = rows.sorted()
    for i in 0..<(sortedRows.count - 1) {
        if sortedRows[i + 1] - sortedRows[i] == 1 {
            return true
        }
    }
    return false
}

// MARK: - Data Augmentation

/// Generate all 8 transformations (rotations and reflections) of a Queens level
public func generateAugmentations(from level: QueensLevel) -> [QueensLevel] {
    var augmentations: [QueensLevel] = []
    let size = level.size
    let regions = level.colorRegions
    
    // Original
    augmentations.append(level)
    
    // 90° rotation
    var rotated90 = Array(repeating: Array(repeating: "", count: size), count: size)
    for r in 0..<size {
        for c in 0..<size {
            rotated90[c][size - 1 - r] = regions[r][c]
        }
    }
    augmentations.append(QueensLevel(size: size, colorRegions: rotated90))
    
    // 180° rotation
    var rotated180 = Array(repeating: Array(repeating: "", count: size), count: size)
    for r in 0..<size {
        for c in 0..<size {
            rotated180[size - 1 - r][size - 1 - c] = regions[r][c]
        }
    }
    augmentations.append(QueensLevel(size: size, colorRegions: rotated180))
    
    // 270° rotation
    var rotated270 = Array(repeating: Array(repeating: "", count: size), count: size)
    for r in 0..<size {
        for c in 0..<size {
            rotated270[size - 1 - c][r] = regions[r][c]
        }
    }
    augmentations.append(QueensLevel(size: size, colorRegions: rotated270))
    
    // Horizontal flip
    var flippedH = Array(repeating: Array(repeating: "", count: size), count: size)
    for r in 0..<size {
        for c in 0..<size {
            flippedH[r][size - 1 - c] = regions[r][c]
        }
    }
    augmentations.append(QueensLevel(size: size, colorRegions: flippedH))
    
    // Vertical flip
    var flippedV = Array(repeating: Array(repeating: "", count: size), count: size)
    for r in 0..<size {
        for c in 0..<size {
            flippedV[size - 1 - r][c] = regions[r][c]
        }
    }
    augmentations.append(QueensLevel(size: size, colorRegions: flippedV))
    
    // Diagonal flip (transpose)
    var transposed = Array(repeating: Array(repeating: "", count: size), count: size)
    for r in 0..<size {
        for c in 0..<size {
            transposed[c][r] = regions[r][c]
        }
    }
    augmentations.append(QueensLevel(size: size, colorRegions: transposed))
    
    // Anti-diagonal flip
    var antiTransposed = Array(repeating: Array(repeating: "", count: size), count: size)
    for r in 0..<size {
        for c in 0..<size {
            antiTransposed[size - 1 - c][size - 1 - r] = regions[r][c]
        }
    }
    augmentations.append(QueensLevel(size: size, colorRegions: antiTransposed))
    
    return augmentations
}

// MARK: - Level Loading

/// Parse a TypeScript level file content and extract the Queens level data
public func parseQueensLevel(from content: String) -> QueensLevel? {
    // Extract size
    guard let sizeMatch = content.range(of: #"size:\s*(\d+)"#, options: .regularExpression),
          let sizeStr = content[sizeMatch].split(separator: ":").last?.trimmingCharacters(in: .whitespaces.union(CharacterSet(charactersIn: ","))),
          let size = Int(sizeStr) else {
        return nil
    }
    
    // Extract colorRegions array
    guard let regionsStart = content.range(of: "colorRegions: ["),
          let regionsEnd = content.range(of: "],", range: regionsStart.upperBound..<content.endIndex) else {
        return nil
    }
    
    let regionsContent = String(content[regionsStart.upperBound..<regionsEnd.lowerBound])
    
    // Parse the 2D array
    var colorRegions: [[String]] = []
    let lines = regionsContent.components(separatedBy: .newlines)
    
    for line in lines {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("[") && trimmed.hasSuffix("],") {
            // Extract content between brackets
            let content = String(trimmed.dropFirst().dropLast(2))
            let elements = content.components(separatedBy: ",").map { element in
                element.trimmingCharacters(in: .whitespaces)
                    .trimmingCharacters(in: CharacterSet(charactersIn: "\""))
            }
            colorRegions.append(elements)
        }
    }
    
    guard colorRegions.count == size else {
        return nil
    }
    
    return QueensLevel(size: size, colorRegions: colorRegions)
}

/// Load all Queens levels from the queens_levels directory
public func loadQueensLevels(from directory: String) -> [QueensLevel] {
    let fileManager = FileManager.default
    var levels: [QueensLevel] = []
    
    guard let files = try? fileManager.contentsOfDirectory(atPath: directory) else {
        print("Could not read directory: \(directory)")
        return levels
    }
    
    let levelFiles = files.filter { $0.hasSuffix(".json") && $0.hasPrefix("level") }.sorted()
    
    for file in levelFiles {
        let filePath = "\(directory)/\(file)"
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: filePath)) else {
            continue
        }
        
        // Parse JSON
        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let size = json["size"] as? Int,
           let colorRegions = json["colorRegions"] as? [[String]],
           colorRegions.count == size {
            levels.append(QueensLevel(size: size, colorRegions: colorRegions))
        }
    }
    
    print("Loaded \(levels.count) Queens levels")
    return levels
}

// MARK: - Queens Solver

/// Simple backtracking solver for Queens puzzles (for generating solutions)
public func solveQueens(level: QueensLevel) -> [[Int]]? {
    let size = level.size
    let regions = level.regionToNumeric()
    var board = Array(repeating: Array(repeating: 0, count: size), count: size)
    
    if solveQueensRecursive(board: &board, row: 0, size: size, regions: regions) {
        return board
    }
    return nil
}

private func solveQueensRecursive(
    board: inout [[Int]], 
    row: Int, 
    size: Int, 
    regions: [[Int]]
) -> Bool {
    if row == size {
        return checkWinCondition(board: board, size: size, colorRegions: regions)
    }
    
    for col in 0..<size {
        if isSafeToPlaceQueen(board: board, row: row, col: col, size: size, colorRegions: regions) {
            board[row][col] = 1
            if solveQueensRecursive(board: &board, row: row + 1, size: size, regions: regions) {
                return true
            }
            board[row][col] = 0
        }
    }
    
    return false
}

// MARK: - Visualization

/// Create a string representation of a Queens board
public func queensBoardString(_ board: [[Int]], regions: [[String]]? = nil) -> String {
    let size = board.count
    var result = ""
    
    // Top border
    result += "+" + String(repeating: "-", count: size * 2 + 1) + "+\n"
    
    for r in 0..<size {
        result += "|"
        for c in 0..<size {
            let queen = board[r][c] == 1 ? "Q" : "."
            result += " \(queen)"
        }
        result += " |\n"
    }
    
    // Bottom border
    result += "+" + String(repeating: "-", count: size * 2 + 1) + "+\n"
    
    if let regions = regions {
        result += "\nRegions:\n"
        for r in 0..<size {
            for c in 0..<size {
                result += regions[r][c]
            }
            result += "\n"
        }
    }
    
    return result
}
