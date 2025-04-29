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

  for (int k = 0; k < temp.size(); ++k) {
    arr[left + k] = temp[k];
  }
}

void seqMerge(vector<int> &arr, int left, int right) {
  if (left >= right)
    return;

  int mid = (left + right) / 2;
  seqMerge(arr, left, mid);
  seqMerge(arr, mid + 1, right);
  merge(arr, left, mid, right);
}

void parallelMerge(vector<int> &arr, int left, int right) {
  if (left >= right)
    return;

  int mid = (left + right) / 2;
#pragma omp parallel sections
  {
#pragma omp section
    parallelMerge(arr, left, mid);
#pragma omp section
    parallelMerge(arr, mid + 1, right);
  }
  merge(arr, left, mid, right);
}

int main() {
  const int N = 50000;
  vector<int> arr(N);
  for (int i = 0; i < N - 1; ++i)
    arr[i] = N - i;

  vector<int> sarr = arr;
  vector<int> parr = arr;

  double start, end;

  start = omp_get_wtime();
  seqMerge(sarr, 0, N - 1);
  end = omp_get_wtime();
  cout << "Seq Time: " << end - start << " seconds " << endl;

  start = omp_get_wtime();
  parallelMerge(parr, 0, N - 1);
  end = omp_get_wtime();
  cout << "Parallel Time: " << end - start << " seconds " << endl;
}
