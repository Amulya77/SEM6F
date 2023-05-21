#include <iostream>
#include <string>
using namespace std;

int main() {
    string sentence, searchString, replaceString;
  
  // Prompt the user to enter a string
    cout << "Enter a string: ";
    getline(cin, sentence);
    
    // Prompt the user to enter the word to replace
    cout << "Enter the word to replace: ";
    cin >> searchString;
    
    // Prompt the user to enter the replacement word
    cout << "Enter the replacement word: ";
    cin >> replaceString;

    // string sentence = "MY name is AMULYA";
    // string searchString = "AMULYA";
    // string replaceString = "AAKARSHA";

    int i = 0, j = 0;
    int start = -1;

    while (sentence[i] != '\0') {
        if (sentence[i] == searchString[j]) {
            if (j == 0) {
                start = i; // mark the starting index of the first occurrence of the searchString
            }
            j++;
            if (j == searchString.length()) {
                break; // we have found the complete searchString, exit the loop
            }
        } else {
            j = 0; // reset the counter if there is a mismatch
            if (start != -1) {
                i = start + 1; // start searching from the next index after the previous mismatch
                start = -1; // reset the starting index
            }
        }

        i++;
    }

    if (start == -1) {
        cout << "Search string not found" << endl;
        return 0;
    }

   // cout << "Starting index: " << start << endl;
    
    string o = "";
    for (int k = 0; k < start; k++) {
        o += sentence[k];
    }

    for (int k = 0; k < replaceString.length(); k++) {
        o += replaceString[k];
    }

    int len = replaceString.length();
    for (int k = start + searchString.length(); k < sentence.length(); k++) {
        o += sentence[k];
    }

    cout << "Modified sentence: " << o << endl;

    return 0;
}
