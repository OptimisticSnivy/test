#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

// Graph as adjacency list
vector<vector<int>> graph;

void parallelBFS(int start) {
    vector<bool> visited(graph.size(), false);
    queue<int> q;

    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int levelSize = q.size();
        vector<int> nextLevel;

        // Process all nodes in the current level in parallel
        #pragma omp parallel for
        for (int i = 0; i < levelSize; ++i) {
            int u;
            // Critical section to pop from queue
            #pragma omp critical
            {
                u = q.front();
                q.pop();
            }
            cout << "Visited " << u << endl;

            // Expand neighbors
            for (auto v : graph[u]) {
                bool alreadyVisited;
                #pragma omp critical
                {
                    alreadyVisited = visited[v];
                    if (!visited[v]) visited[v] = true;
                }
                if (!alreadyVisited) {
                    #pragma omp critical
                    {
                        nextLevel.push_back(v);
                    }
                }
            }
        }

        // Push next level into queue
        for (auto v : nextLevel) {
            q.push(v);
        }
    }
}

int main() {
    int n = 6;
    graph.resize(n);
    // Example graph
    graph[0] = {1, 2};
    graph[1] = {3, 4};
    graph[2] = {4};
    graph[3] = {5};
    graph[4] = {5};
    graph[5] = {};

    cout << "Parallel BFS starting from node 0:" << endl;
    parallelBFS(0);

    return 0;
}
