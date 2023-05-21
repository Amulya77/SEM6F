#include <iostream>
//Code By: AMULYA MAURYA
//2023-05-21
#include<bits/stdc++.h>
using namespace std;


struct Node {
    char data;
    Node* next;
};

bool characterExists(Node* head, char c) {
    Node* current = head;
    while (current != nullptr) {
        if (current->data == c) {
            return true;
        }
        current = current->next;
    }
    return false;
}

int main() {
    // Create a linked list
    Node* head = new Node {'A', new Node {'B', new Node {'C', nullptr}}};

    // Check if character 'B' exists in the linked list
    char characterToCheck = 'B';
    bool exists = characterExists(head, characterToCheck);

    if (exists) {
        cout << "Character '" << characterToCheck << "' exists in the linked list." << endl;
    } else {
        cout << "Character '" << characterToCheck << "' does not exist in the linked list." << endl;
    }

    // Clean up memory (deallocate nodes)
    Node* current = head;
    while (current != nullptr) {
        Node* temp = current;
        current = current->next;
        delete temp;
    }

    return 0;
}
