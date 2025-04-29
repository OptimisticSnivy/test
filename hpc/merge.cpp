#include <iostream>
#include <omp.h>
#include <vector>
using namespace std;

void merge(vector<int> &arr, int left, int mid, int right) {
  vector<int> temp;
  int i = left, j = mid + 1;

  while (i <= mid && j <= right) {
    if (arr[i] <= arr[j])
      temp.push_back(arr[i++]);
    else
      temp.push_back(arr[j++]);
  }

  while (i <= mid)
    temp.push_back(arr[i++]);
  while (j <= right)
    temp.push_back(arr[j++]);

  for (int k = 0; k < temp.size(); ++k)
    arr[left + k] = temp[k];
}

void mergeSortSequential(vector<int> &arr, int left, int right) {
  if (left >= right)
    return;

  int mid = (left + right) / 2;
  mergeSortSequential(arr, left, mid);
  mergeSortSequential(arr, mid + 1, right);
  merge(arr, left, mid, right);
}

void mergeSortParallel(vector<int> &arr, int left, int right) {
  if (left >= right)
    return;

  int mid = (left + right) / 2;

#pragma omp parallel sections
  {
#pragma omp section
    mergeSortParallel(arr, left, mid);
#pragma omp section
    mergeSortParallel(arr, mid + 1, right);
  }

  merge(arr, left, mid, right);
}

int main() {
  const int N = 50000;
  vector<int> original(N);
  for (int i = 0; i < N; ++i)
    original[i] = N - i; // reverse order (worst case)

  vector<int> seqArr = original;
  vector<int> parArr = original;

  double start, end;

  // Sequential timing
  start = omp_get_wtime();
  mergeSortSequential(seqArr, 0, N - 1);
  end = omp_get_wtime();
  cout << "Sequential Time: " << end - start << " seconds\n";

  // Parallel timing
  start = omp_get_wtime();
  mergeSortParallel(parArr, 0, N - 1);
  end = omp_get_wtime();
  cout << "Parallel Time:   " << end - start << " seconds\n";

  return 0;
}
