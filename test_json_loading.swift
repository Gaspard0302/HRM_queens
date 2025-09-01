import Foundation

// Simple test to verify JSON level loading
func testLoadQueensLevels() {
    let directory = "queens_levels_json"
    
    // Check if directory exists
    let fileManager = FileManager.default
    guard fileManager.fileExists(atPath: directory) else {
        print("Error: Directory \(directory) not found")
        return
    }
    
    // Get all JSON files
    guard let files = try? fileManager.contentsOfDirectory(atPath: directory) else {
        print("Error: Could not read directory contents")
        return
    }
    
    let levelFiles = files.filter { $0.hasSuffix(".json") && $0.hasPrefix("level") }.sorted()
    print("Found \(levelFiles.count) level files")
    
    var successCount = 0
    var failedFiles: [String] = []
    
    for file in levelFiles {
        let filePath = "\(directory)/\(file)"
        
        if let data = try? Data(contentsOf: URL(fileURLWithPath: filePath)),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let size = json["size"] as? Int,
           let colorRegions = json["colorRegions"] as? [[String]],
           colorRegions.count == size {
            successCount += 1
        } else {
            failedFiles.append(file)
        }
    }
    
    print("\nResults:")
    print("Successfully loaded: \(successCount)/\(levelFiles.count) levels")
    
    if !failedFiles.isEmpty {
        print("\nFailed to load:")
        for file in failedFiles {
            print("  - \(file)")
        }
    }
    
    // Try to load and display first level as example
    if let firstFile = levelFiles.first {
        print("\nExample - Loading \(firstFile):")
        let filePath = "\(directory)/\(firstFile)"
        
        if let data = try? Data(contentsOf: URL(fileURLWithPath: filePath)),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let size = json["size"] as? Int,
           let colorRegions = json["colorRegions"] as? [[String]] {
            print("  Size: \(size)")
            print("  Color regions count: \(colorRegions.count)")
            print("  First row: \(colorRegions[0])")
        }
    }
}

// Run the test
testLoadQueensLevels()
