#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>
#include <set>

using namespace std;

void parallelDFS(int start_node, const vector<vector<int>>& adj) {
    int num_nodes = adj.size();
    vector<bool> visited(num_nodes, false);
    stack<int> s;

    s.push(start_node);
    visited[start_node] = true;
    cout << "Visited: " << start_node << endl;

    while (!s.empty()) {
        int u = s.top();
        s.pop();

        vector<int> neighbors = adj[u];
        #pragma omp parallel for
        for (int i = 0; i < neighbors.size(); ++i) {
            int v = neighbors[i];
            #pragma omp critical // Protect shared 'visited' and 's'
            {
                if (!visited[v]) {
                    visited[v] = true;
                    cout << "Visited: " << v << " (from " << u << ")" << endl;
                    s.push(v);
                }
            }
        }
    }
}

int main() {
    // Example graph (adjacency list)
    vector<vector<int>> adj = {
        {1, 2},
        {0, 3, 4},
        {0, 5},
        {1},
        {1, 6},
        {2},
        {4}
    };

    parallelDFS(0, adj);

    return 0;
}
