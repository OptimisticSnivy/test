#include <iostream>
#include <omp.h>
#include <set>
#include <stack>
#include <vector>

using namespace std;

void parallelDFS(int start, const vector<vector<int>> &adj) {
  int n = adj.size();
  vector<bool> visited(n, false);
  stack<int> s;

  s.push(start);
  visited[start] = true;
  cout << "Visited: " << start << endl;

  while (!s.empty()) {
    int u = s.top();
    s.pop();

    vector<int> neighbors = adj[u];
    #pragma omp parallel for
    for (int i = 0;i< neighbors.size(); ++i) {
      int v = neighbors[i];
      #pragma omp critical
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
  vector<vector<int>> adj = {{1, 2}, {0, 3, 4}, {0, 5}, {1}, {1, 6}, {2}, {4}};

  parallelDFS(0, adj);

  return 0;
}
