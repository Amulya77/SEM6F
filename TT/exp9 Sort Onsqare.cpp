//Code By: AMULYA MAURYA
//2023-04-07
#include<bits/stdc++.h>
using namespace std;

struct Node {
    int data;
    Node* next;
};
void printList(Node* head) {
    while (head != nullptr) {
        cout << head->data << " ";
        head = head->next;
    }
    cout << endl;
}


void swap(Node* a, Node* b) {
    int temp = a->data;
    a->data = b->data;
    b->data = temp;
}
int sizeOfLinkedList(Node* head) {
    int size = 0;
    Node* current = head;
    while (current != nullptr) {
        size++;
        current = current->next;
    }
    return size;
}


void bubbleSort(Node* head) {
    if (head == nullptr) return;

    Node* ptr1;
    Node* ptr2 = nullptr;
    int size=sizeOfLinkedList(head);
    for (int i = 0; i < size; ++i) {
        ptr1 = head;

        for (int j = 0; j < size - i - 1; ++j) {
            if (ptr1->data > ptr1->next->data) {
                swap(ptr1, ptr1->next);
            }
            ptr1 = ptr1->next;
        }
        ptr2 = ptr1;
    }
}



int main() {
    Node* head = new Node{5, new Node{3, new Node{8, new Node{1, new Node{2, nullptr}}}}};


    cout<<"Code By Amulya "<<endl;
    cout << "Before sorting: ";
    printList(head);
    bubbleSort(head);
    cout << "After sorting: ";
    printList(head);
    return 0;
}