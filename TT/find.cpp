#include <iostream>
#include <string>

using namespace std;

int main() {
  string input, word, replacement;
  
  // Prompt the user to enter a string
  cout << "Enter a string: ";
  getline(cin, input);
  
  // Prompt the user to enter the word to replace
  cout << "Enter the word to replace: ";
  cin >> word;
  
  // Prompt the user to enter the replacement word
  cout << "Enter the replacement word: ";
  cin >> replacement;
  
  // Find and replace the word in the input string
  string output;
  int pos = 0, word_len = word.length();
  while (pos < input.length()) {
    int found_pos = -1;
    for (int i = pos; i < input.length() - word_len + 1; i++) {
      bool match = true;
      for (int j = 0; j < word_len; j++) {
        if (input[i + j] != word[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        found_pos = i;
        break;
      }
    }
    if (found_pos == -1) {
      output += input.substr(pos);
      break;
    }
    output += input.substr(pos, found_pos - pos) + replacement;
    pos = found_pos + word_len;
  }
  
  // Output the modified string
  cout << "Modified string: " << output << endl;
  
  return 0;
}