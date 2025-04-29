#include <climits>
#include <iostream>
#include <limits>
#include <omp.h>
#include <vector>

using namespace std;

int findMin(const vector<int> &arr) {
  int minVal = INT_MAX;

#pragma omp parallel for reduction(min : minVal)
  for (int i = 0; i < arr.size(); ++i) {
    if (arr[i] < minVal) {
      minVal = arr[i];
    }
  }
  return minVal;
}

int findMax(const vector<int> &arr) {
  int maxVal = INT_MIN;

#pragma omp parallel for reduction(max : maxVal)
  for (int i = 0; i < arr.size(); ++i) {
    if (arr[i] > maxVal) {
      maxVal = arr[i];
    }
  }
  return maxVal;
}

int findSum(const vector<int> &arr) {
  int sum = 0;

#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < arr.size(); ++i) {
    sum += arr[i];
  }
  return sum;
}

double findAverage(const vector<int> &arr) {
  int sum = findSum(arr);
  return static_cast<double>(sum) / arr.size();
}

int main() {
  vector<int> arr = {12, 45, 23, 67, 34, 89, 2, 99, 5, 31};

  cout << "Array: ";
  for (int x : arr)
    cout << x << " ";
  cout << "\n\n";

  int minVal = findMin(arr);
  int maxVal = findMax(arr);
  int sum = findSum(arr);
  double average = findAverage(arr);

  cout << "Parallel Min: " << minVal << endl;
  cout << "Parallel Max: " << maxVal << endl;
  cout << "Parallel Sum: " << sum << endl;
  cout << "Parallel Average: " << average << endl;

  return 0;
}
