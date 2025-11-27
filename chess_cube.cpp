#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <map>

// A struct to hold the data for a single location in the Memory Palace
struct CubeCell {
    int x, y, z;
    int color_parity; // 0 for White, 1 for Black (Property A/B)
    std::string loc_prefix_mapped;
};

class ChessCubeLattice {
private:
    int size;
    std::vector<CubeCell> cells;

public:
    ChessCubeLattice(int s = 8) : size(s) {
        generate_coordinates();
    }

    // Generates the 512 coordinates with Parity Check
    void generate_coordinates() {
        cells.reserve(size * size * size);
        for (int x = 1; x <= size; ++x) {
            for (int y = 1; y <= size; ++y) {
                for (int z = 1; z <= size; ++z) {
                    int parity = (x + y + z) % 2;
                    cells.push_back({x, y, z, parity, ""});
                }
            }
        }
        std::cout << "Lattice Generated: " << cells.size() << " unique locations." << std::endl;
    }

    // The Path Loss Distance Metric (Euclidean)
    double calculate_distance(const CubeCell& a, const CubeCell& b) {
        return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2) + std::pow(a.z - b.z, 2));
    }

    // Example: Find all cells with specific parity (e.g., for Separation Loss check)
    std::vector<CubeCell> get_cells_by_parity(int parity) {
        std::vector<CubeCell> result;
        for (const auto& cell : cells) {
            if (cell.color_parity == parity) {
                result.push_back(cell);
            }
        }
        return result;
    }
};

int main() {
    std::cout << "--- C++ Chess Cube Engine Initialized ---" << std::endl;
    
    ChessCubeLattice palace(8);
    
    // Example Usage
    CubeCell start = {1, 1, 1, 1, ""};
    CubeCell end = {8, 8, 8, 0, ""};
    
    double dist = palace.calculate_distance(start, end);
    std::cout << "Distance from (1,1,1) to (8,8,8): " << dist << std::endl;
    
    return 0;
}
