#ifndef CHATROOM_SERVER_H
#define CHATROOM_SERVER_H

#include <string>
#include <stdlib.h>
#include <netinet/in.h>
#include <list>

#include "common.hpp"

using namespace std;
class Server
{
public:
    Server();
    ~Server();
    void Init();
    void Close();
    void Start();
    //int send2AllMsg(int clientfd);
private:
    //广播消息
    int send2AllMsg(int clientfd);    
    sockaddr_in serverAddr;
    int listener;
    int epfd;
    list<int> clients_list;
};


#endif /* server_hpp */

