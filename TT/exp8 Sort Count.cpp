#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

void countingSort(vector<int>& arr) {
    int minVal = *min_element(arr.begin(), arr.end());
    int maxVal = *max_element(arr.begin(), arr.end());
    int range = maxVal - minVal + 1;

    // Create a count array to store the count of each element
    vector<int> count(range, 0);

    // Count the occurrences of each element
    for (int num : arr) {
        count[num - minVal]++;
    }

    // Modify the count array to store the cumulative count
    for (int i = 1; i < range; i++) {
        count[i] += count[i - 1];
    }

    // Reconstruct the sorted array based on the counts
    vector<int> sorted(arr.size());
    for (int i = arr.size() - 1; i >= 0; i--) {
        int num = arr[i];
        int index = count[num - minVal] - 1;
        sorted[index] = num;
        count[num - minVal]--;
    }

    // Update the original array with the sorted values
    arr = sorted;
}

int main() {
    vector<int> arr = {9, -5, 7, 1, -8, 2, 4};

    cout << "Before sorting: ";
    for (int num : arr) {
        cout << num << " ";
    }
    cout << endl;

    countingSort(arr);

    cout << "After sorting: ";
    for (int num : arr) {
        cout << num << " ";
    }
    cout << endl;

    return 0;
}
