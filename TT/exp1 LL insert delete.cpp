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
        int val=this->data;
        if(this->next!=NULL)
        delete next;
        next=NULL;
    }
    
};


void inserthed(Node *&head, int data){
    Node*temp=new Node(data);
    temp->next=head;
    head=temp;
}

// void inserttail(Node*&head,int data){
//     Node*temp=new Node(data);
//     Node *curr=head;
//     while(curr->next!=NULL){
//         curr=curr->next;
//     }
//     curr->next=temp;
// }
void insertattailOP(Node*&tail,int data){
    Node *temp =new Node(data);
    tail->next=temp;
    tail=temp;
}

void insertatPos(Node*&head,Node*&tail,int pos , int data){
    Node*temp=new Node(data);
    if(pos==1){
        inserthed(head,data);

        // temp->next=head;
        // head=temp;
        return;
    }

    Node *curr=head;
    int i=1;
    while(i<pos-1){
        curr=curr->next;
        i++;
    }
    
    if(curr->next==NULL){
        insertattailOP(tail,data);
        return;
    }

    temp->next=curr->next;
    curr->next=temp;

}

void print(Node* &head){
    Node*temp=head;
    while(temp!=NULL){
        cout<<temp->data<<" ";
        temp=temp->next;
    }
    cout<<endl;
}


void deleteNode(int pos, Node*&head){
    if(pos==1){
        Node*temp=head;
        head=head->next;
        temp->next=NULL;
        delete temp;
    }
    else{

        Node*curr=head;
        Node*prev=NULL;
        int i=1;
        while(i<pos){
            prev=curr;
            curr=curr->next;
            i++;
        }

        prev->next=curr->next;
        curr->next=NULL;
        delete curr;
    }

}




int main()
{
    Node* node1=new Node(10);
    
    Node*head=node1;
    Node*tail=node1;
    print(head);
    inserthed(head,40);
    print(head);

    //inserttail(head,50);
    //print(head);
    insertattailOP(tail,60);
    print(head);

    insertatPos(head,tail,3,12);
    print(head);

    inserthed(head,11);
    inserthed(head,233);
    print(head);
    deleteNode(3,head);
    print(head);
    return 0;
}