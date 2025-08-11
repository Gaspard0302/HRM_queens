//
//  Sudoku.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import Foundation

enum Difficulty {
    case veryEasy
    case easy
    case medium
    case hard
    case extreme
}

func generateSudoku(difficulty: Difficulty) -> (puzzle: [[Int]], solution: [[Int]]) {
    var board = Array(repeating: Array(repeating: 0, count: 9), count: 9)

    _ = fillGrid(&board)

    let solution = board

    let targetClues: ClosedRange<Int> = {
        switch difficulty {
        case .veryEasy: return 46...50
        case .easy: return 40...45
        case .medium: return 32...39
        case .hard: return 28...31
        case .extreme: return 17...27
        }
    }()

    var puzzle = board
    var cells = Array(0..<81)
    cells.shuffle()

    var cursor = 0
    var clues = 81

    while cursor < cells.count && clues > targetClues.upperBound {
        let idx = cells[cursor]
        cursor += 1
        let r = idx / 9
        let c = idx % 9
        let backup = puzzle[r][c]
        puzzle[r][c] = 0

        var test = puzzle
        var solutionCounter = 0
        _ = solve(&test, count: &solutionCounter, stopAfter: 2)
        if solutionCounter != 1 {
            puzzle[r][c] = backup
        } else {
            clues -= 1
        }
    }

    if clues > targetClues.lowerBound {
        for j in cursor..<cells.count {
            if clues <= targetClues.lowerBound { break }
            let idx = cells[j]
            let r = idx / 9
            let c = idx % 9
            let backup = puzzle[r][c]
            puzzle[r][c] = 0

            var test = puzzle
            var solutionCounter = 0
            _ = solve(&test, count: &solutionCounter, stopAfter: 2)
            if solutionCounter != 1 {
                puzzle[r][c] = backup
            } else {
                clues -= 1
            }
        }
    }

    return (puzzle, solution)
}

@inline(__always)
private func bit(for value: Int) -> Int { 1 << (value - 1) }

@inline(__always)
private func boxIndex(forRow row: Int, col: Int) -> Int { (row / 3) * 3 + (col / 3) }

@inline(__always)
private func buildMasks(from grid: [[Int]]) -> (rows: [Int], cols: [Int], boxes: [Int]) {
    var rows = Array(repeating: 0, count: 9)
    var cols = Array(repeating: 0, count: 9)
    var boxes = Array(repeating: 0, count: 9)

    for r in 0..<9 {
        var rowMask = 0
        for c in 0..<9 {
            let v = grid[r][c]
            if v != 0 {
                let b = bit(for: v)
                rowMask |= b
                cols[c] |= b
                boxes[boxIndex(forRow: r, col: c)] |= b
            }
        }
        rows[r] = rowMask
    }
    return (rows, cols, boxes)
}

private let allDigits: [Int] = [1, 2, 3, 4, 5, 6, 7, 8, 9]

@discardableResult
private func fillGrid(_ grid: inout [[Int]]) -> Bool {
    var (rows, cols, boxes) = buildMasks(from: grid)
    return fillGridRec(&grid, &rows, &cols, &boxes)
}

@inline(__always)
private func fillGridRec(
    _ grid: inout [[Int]],
    _ rows: inout [Int],
    _ cols: inout [Int],
    _ boxes: inout [Int]
) -> Bool {
    guard let (row, col) = firstEmptyCell(in: grid) else { return true }

    var numbers = allDigits
    numbers.shuffle()

    let bIdx = boxIndex(forRow: row, col: col)
    let used = rows[row] | cols[col] | boxes[bIdx]

    for num in numbers {
        let b = bit(for: num)
        if (used & b) != 0 { continue }

        grid[row][col] = num
        rows[row] |= b
        cols[col] |= b
        boxes[bIdx] |= b

        if fillGridRec(&grid, &rows, &cols, &boxes) { return true }

        grid[row][col] = 0
        rows[row] &= ~b
        cols[col] &= ~b
        boxes[bIdx] &= ~b
    }
    return false
}

@discardableResult
private func solve(
    _ grid: inout [[Int]],
    count solutions: inout Int,
    stopAfter limit: Int
) -> Bool {
    var (rows, cols, boxes) = buildMasks(from: grid)
    return solveRec(&grid, &solutions, limit, &rows, &cols, &boxes)
}

@inline(__always)
private func solveRec(
    _ grid: inout [[Int]],
    _ solutions: inout Int,
    _ limit: Int,
    _ rows: inout [Int],
    _ cols: inout [Int],
    _ boxes: inout [Int]
) -> Bool {
    if solutions >= limit { return true }
    guard let (row, col) = firstEmptyCell(in: grid) else {
        solutions += 1
        return solutions >= limit
    }

    let bIdx = boxIndex(forRow: row, col: col)
    let used = rows[row] | cols[col] | boxes[bIdx]

    for num in 1...9 {
        let b = bit(for: num)
        if (used & b) != 0 { continue }

        grid[row][col] = num
        rows[row] |= b
        cols[col] |= b
        boxes[bIdx] |= b

        if solveRec(&grid, &solutions, limit, &rows, &cols, &boxes) { return true }

        grid[row][col] = 0
        rows[row] &= ~b
        cols[col] &= ~b
        boxes[bIdx] &= ~b
    }
    return false
}

@inline(__always)
private func isValid(_ value: Int, atRow row: Int, col: Int, in grid: [[Int]]) -> Bool {
    for i in 0..<9 {
        if grid[row][i] == value || grid[i][col] == value { return false }
    }
    let boxRow = (row / 3) * 3
    let boxCol = (col / 3) * 3
    for r in boxRow..<(boxRow + 3) {
        let rowArr = grid[r]
        for c in boxCol..<(boxCol + 3) where rowArr[c] == value {
            return false
        }
    }
    return true
}

@inline(__always)
private func firstEmptyCell(in grid: [[Int]]) -> (Int, Int)? {
    for r in 0..<9 {
        let rowArr = grid[r]
        for c in 0..<9 where rowArr[c] == 0 {
            return (r, c)
        }
    }
    return nil
}

@inline(__always)
private func clueCount(in grid: [[Int]]) -> Int {
    var count = 0
    for r in 0..<9 {
        for c in 0..<9 where grid[r][c] != 0 {
            count &+= 1
        }
    }
    return count
}

func sudokuBoardString(_ board: [[Int]]) -> String {
    let horizontalLine = "+-------+-------+-------+"
    var result = horizontalLine + "\n"

    for (rowIndex, row) in board.enumerated() {
        var line = "|"

        for (colIndex, cell) in row.enumerated() {
            let displayValue = cell == 0 ? "." : "\(cell)"
            line += " \(displayValue)"

            if (colIndex + 1) % 3 == 0 {
                line += " |"
            }
        }

        result += line + "\n"

        if (rowIndex + 1) % 3 == 0 {
            result += horizontalLine + "\n"
        }
    }

    return result.trimmingCharacters(in: .whitespacesAndNewlines)
}
