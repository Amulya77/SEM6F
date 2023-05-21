#include <iostream>
#include <unordered_map>

using namespace std;

class TrieNode {
public:
    unordered_map<char, TrieNode*> children;
    bool isEndOfWord;

    TrieNode() {
        isEndOfWord = false;
    }
};

class Trie {
private:
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    void insert(string word) {
        TrieNode* current = root;
        for (char ch : word) {
            if (current->children.find(ch) == current->children.end()) {
                current->children[ch] = new TrieNode();
            }
            current = current->children[ch];
        }
        current->isEndOfWord = true;
    }

    bool search(string word) {
        TrieNode* current = root;
        for (char ch : word) {
            if (current->children.find(ch) == current->children.end()) {
                return false;
            }
            current = current->children[ch];
        }
        return current->isEndOfWord;
    }
};

int main() {
    Trie trie;
    trie.insert("apple");
    trie.insert("banana");
    trie.insert("orange");

    cout << "Search results:" << endl;
    cout << "apple: " << (trie.search("apple") ? "Found" : "Not found") << endl;
    cout << "banana: " << (trie.search("banana") ? "Found" : "Not found") << endl;
    cout << "orange: " << (trie.search("orange") ? "Found" : "Not found") << endl;
    cout << "grape: " << (trie.search("grape") ? "Found" : "Not found") << endl;

    return 0;
}
