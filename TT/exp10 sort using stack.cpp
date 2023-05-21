#include <iostream>
#include <vector>
#include <stack>

using namespace std;

void reverseArrays(vector<int>& arr1, vector<int>& arr2) {
    stack<int> stack1, stack2;

    // Push elements of arr1 onto stack1
    for (int num : arr1) {
        stack1.push(num);
    }

    // Push elements of arr2 onto stack2
    for (int num : arr2) {
        stack2.push(num);
    }

    // Pop elements from stack1 into arr1 in reverse order
    for (int i = 0; i < arr1.size(); i++) {
        arr1[i] = stack1.top();
        stack1.pop();
    }

    // Pop elements from stack2 into arr2 in reverse order
    for (int i = 0; i < arr2.size(); i++) {
        arr2[i] = stack2.top();
        stack2.pop();
    }
}

int main() {
    vector<int> arr1 = {90, 80, 70, 60, 50};
    vector<int> arr2 = {95, 85, 75, 65, 55};

    cout << "Before reversing:" << endl;
    cout << "arr1: ";
    for (int num : arr1) {
        cout << num << " ";
    }
    cout << endl;

    cout << "arr2: ";
    for (int num : arr2) {
        cout << num << " ";
    }
    cout << endl;

    reverseArrays(arr1, arr2);

    cout << "After reversing:" << endl;
    cout << "arr1: ";
    for (int num : arr1) {
        cout << num << " ";
    }
    cout << endl;

    cout << "arr2: ";
    for (int num : arr2) {
        cout << num << " ";
    }
    cout << endl;

    return 0;
}
