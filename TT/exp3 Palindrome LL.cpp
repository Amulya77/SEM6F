//Code By: AMULYA MAURYA
//2023-05-21
#include<bits/stdc++.h>
using namespace std;


class Node{
    public:
    int data;
    Node *next;
    Node(int x){
        this->data=x;
        this->next=NULL;
}
 ~Node(){
        int v=this->data;
        if(this->next!=NULL){
            delete next;
            this->next=NULL;

        }
        cout<<"Memory is free for node with data "<<v<<endl;
    }
};

Node* reverse(Node*head){
    Node*prev=NULL;
    Node*curr=head;

    while(curr){
        Node*forward=curr->next;
        curr->next=prev;
        prev=curr;
        curr=forward;
    }
    return prev;
}

Node* middle(Node* head){
    Node* fast= head -> next;
    Node* slow = head;
    while(fast && fast -> next){
        fast = fast -> next -> next;
        slow = slow -> next;
    }
    return slow;
}

bool check_pallindrome(Node* head){
    //edge cases 
    if(!head || !head->next){
        return true;
    }
    Node* mid = middle(head);
    Node* head2 = reverse(mid -> next);
    mid -> next = NULL;
    while(head2){
        if (head->data != head2->data){
            return false;
        }
        head = head->next;
        head2 = head2->next;
    }
    return true;
}

void populate(Node* &head, Node* &tail, char ch){
    Node* new_node = new Node(ch);
    if(!head && !tail){
        head = new_node;
        tail = new_node;
        return;
    }
    
    tail -> next = new_node;
    tail = tail -> next;
}
Node* createList(string s){
    Node* head = NULL;
    Node* tail = NULL; 
    for(int i = 0; i < s.size(); i++){
        char ch= s[i];
        populate(head, tail, ch);
    }
    return head;
}


void print(Node*head){
    Node*curr=head;
    while(curr){
        cout<<curr->data<<" ";
        curr=curr->next;
    }
    cout<<endl;
}

int main()
{
        string s;
        cout << "Enter a string: ";
        cin >> s;
        Node* head = createList(s);
        bool res = check_pallindrome(head);
        if (res){
            cout << "The given list is pallindrome" << endl;
        } else{
            cout << "The given list is not pallindrome" << endl;
        }  
    return 0;
}