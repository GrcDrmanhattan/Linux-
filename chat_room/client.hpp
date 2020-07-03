#ifndef CHATROOM_CLIENT_H
#define CHATROOM_CLIENT_H

#include <string>
#include <sys/types.h>

#include "common.hpp"

using namespace std;

class Client
{
public:
    Client();
    ~Client();
    void Connect();
    void Close();
    //启动客户端
    void Start();

private:
    //server sock
    int sock;
    //
    pid_t pid;
    //
    int epfd;

    //管道，fd[0] 父进程读，fd[1]子进程写
    int pipe_fd[2];
    //是否客户端工作
    bool isClientWork;

    //chat message buffer
    char message[BUF_SIZE];
    //
    sockaddr_in serverAddr;
    
};



#endif

