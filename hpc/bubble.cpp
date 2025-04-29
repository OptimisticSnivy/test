#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

void normalBubble(vector<int> arr) {
  int n = arr.size();

  for (int i = 0; i < n - 1; ++i) {
    for (int j = 0; j < n - i - 1; ++j) {
      if (arr[j] > arr[j + 1]) {
        swap(arr[j], arr[j + 1]);
      }
    }
  }
}

void parallelBubble(vector<int> &arr) {
  int n = arr.size();
  bool isSorted = false;

  while (!isSorted) {
    isSorted = true;

    #pragma omp parallel for
    for (int i = 1; i < n - 1; i += 2) {
      if (arr[i] > arr[i + 1]) {
        swap(arr[i], arr[i + 1]);
        isSorted = false;
      }
    }

    #pragma omp parallel for
    for (int i = 0; i < n - 1; i += 2) {
      if (arr[i] > arr[i + 1]) {
        swap(arr[i], arr[i + 1]);
        isSorted = false;
      }
    }
  }
}

void printArray(vector<int> arr) {
  for (int x : arr)
    cout << x << " ";
}

int main() {
  // vector<int> arr = {64, 34, 25, 12, 22, 11, 90};

  vector<int> arr(7000);
  for (int i = 0; i < 7000; ++i) {
    arr[i] = 7000 - i;
  }
  vector<int> sarr = arr;
  vector<int> parr = arr;

  cout << "Original Array: ";
  // printArray(arr);
  cout << endl;

  // Sequential
  cout << "Sequential: ";
  double sstart = omp_get_wtime();
  normalBubble(sarr);
  double send = omp_get_wtime();
  cout << "Sorted Array: ";
  // printArray(sarr);
  cout << endl;
  cout << "Normal time: " << (send - sstart) << "seconds" << endl;

  // Parallel
  cout << "Parallel: ";
  double pstart = omp_get_wtime();
  parallelBubble(parr);
  double pend = omp_get_wtime();
  cout << "Sorted Array: ";
  // printArray(parr);
  cout << endl;
  cout << "parallel time: " << (pend - pstart) << "seconds" << endl;

  return 0;
}
